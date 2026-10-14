package com.leansoft.draw.mlkitforface

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothGattServer
import android.bluetooth.BluetoothGattServerCallback
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.AdvertiseCallback
import android.bluetooth.le.AdvertiseData
import android.bluetooth.le.AdvertiseSettings
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Rect
import android.os.Build
import android.os.Bundle
import android.os.ParcelUuid
import android.util.Size
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.leansoft.draw.mlkitforface.databinding.ActivityMainBinding
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private val processing = AtomicBoolean(false)
    private var detector: FaceDetector? = null
    private var embeddingModel: FaceEmbeddingModel? = null
    private lateinit var attendanceRepository: AttendanceRepository
    private var cameraProvider: ProcessCameraProvider? = null
    private var gattServer: BluetoothGattServer? = null
    private var advertising = false
    private var connectedDevice: android.bluetooth.BluetoothDevice? = null
    private var latestPayload = "{\"faces\":0,\"status\":\"waiting\"}"
    private var attendancePayloadUntil = 0L
    private var selectedMode = 0

    private val bluetoothManager by lazy { getSystemService(BluetoothManager::class.java) }
    private val adapter: BluetoothAdapter? get() = bluetoothManager?.adapter

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else showMessage("Cần quyền camera để chạy nhận diện trực tiếp.")
    }

    private val blePermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { result ->
        if (result.values.all { it }) startBleServer()
        else showMessage("Cần quyền Nearby devices để phát BLE cho web dashboard.")
    }

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@registerForActivityResult
        try {
            val image = InputImage.fromFilePath(this, uri)
            binding.tvMlResult.text = "Đang chạy model trên ảnh…"
            activeDetector().process(image)
                .addOnSuccessListener { faces -> showFaces(faces, image.width, image.height, "Ảnh") }
                .addOnFailureListener { showMessage("ML Kit: ${it.localizedMessage ?: "không xử lý được ảnh"}") }
        } catch (error: Exception) {
            showMessage("Không đọc được ảnh: ${error.localizedMessage}")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        attendanceRepository = AttendanceRepository(this)
        binding.etSession.setText(attendanceRepository.currentSession())
        latestPayload = JSONObject()
            .put("source", "camera")
            .put("model", "fast")
            .put("faces", 0)
            .put("session", attendanceRepository.currentSession())
            .put("presentCount", attendanceRepository.attendance().count { it.session == attendanceRepository.currentSession() })
            .put("registeredCount", attendanceRepository.students().size)
            .put("attendees", attendanceSnapshot(attendanceRepository.currentSession()))
            .put("updatedAt", System.currentTimeMillis())
            .toString()
        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.tvBleUuid.text = "Service: $SERVICE_UUID   ·   data: $RESULT_UUID   ·   notify: $NOTIFY_UUID"
        binding.spinnerMode.adapter = ArrayAdapter.createFromResource(
            this, R.array.detection_modes, android.R.layout.simple_spinner_dropdown_item
        )
        binding.spinnerMode.setSelection(selectedMode)
        binding.spinnerMode.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                if (selectedMode != position) {
                    selectedMode = position
                    detector?.close()
                    detector = null
                    binding.tvMlResult.text = "Đã chọn model ${modeName()}. Hãy nhìn vào camera hoặc chọn ảnh."
                }
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
        }
        binding.btnCamera.setOnClickListener {
            if (cameraProvider != null) stopCamera() else requestCamera()
        }
        binding.btnPickImage.setOnClickListener { pickImage.launch("image/*") }
        binding.btnAttendanceRecords.setOnClickListener { startActivity(Intent(this, AttendanceActivity::class.java)) }
        binding.btnCheckIn.setOnClickListener { checkInFromCamera() }
        binding.btnBle.setOnClickListener {
            if (advertising) stopBleServer() else requestBlePermissions()
        }
    }

    private fun modeName() = if (selectedMode == 0) "Nhanh" else "Chi tiết"

    private fun activeDetector(): FaceDetector {
        detector?.let { return it }
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(if (selectedMode == 0) FaceDetectorOptions.PERFORMANCE_MODE_FAST else FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(if (selectedMode == 0) FaceDetectorOptions.LANDMARK_MODE_NONE else FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(if (selectedMode == 0) FaceDetectorOptions.CLASSIFICATION_MODE_NONE else FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .enableTracking()
            .setMinFaceSize(0.08f)
            .build()
        return FaceDetection.getClient(options).also { detector = it }
    }

    private fun requestCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else cameraPermission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        binding.tvCameraHint.visibility = android.view.View.GONE
        binding.btnCamera.text = "Tắt camera"
        binding.btnCheckIn.isEnabled = true
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                cameraProvider = provider
                val preview = Preview.Builder().build().also { it.surfaceProvider = binding.previewView.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(cameraExecutor) { proxy -> analyzeFrame(proxy) }
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
                binding.tvMlResult.text = "Model ${modeName()} sẵn sàng · đưa khuôn mặt vào khung hình"
            } catch (error: Exception) {
                cameraProvider = null
                binding.tvCameraHint.visibility = android.view.View.VISIBLE
                binding.tvCameraHint.text = "Không khởi động được camera: ${error.localizedMessage}"
                binding.btnCamera.text = "Bật camera"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    private fun analyzeFrame(proxy: ImageProxy) {
        val mediaImage = proxy.image
        if (mediaImage == null || !processing.compareAndSet(false, true)) {
            proxy.close()
            return
        }
        val input = InputImage.fromMediaImage(mediaImage, proxy.imageInfo.rotationDegrees)
        activeDetector().process(input)
            .addOnSuccessListener { faces ->
                val rotated = proxy.imageInfo.rotationDegrees == 90 || proxy.imageInfo.rotationDegrees == 270
                val w = if (rotated) proxy.height else proxy.width
                val h = if (rotated) proxy.width else proxy.height
                runOnUiThread { showFaces(faces, w, h, "Camera") }
            }
            .addOnFailureListener { error -> runOnUiThread { showMessage("ML Kit: ${error.localizedMessage}") } }
            .addOnCompleteListener {
                processing.set(false)
                proxy.close()
            }
    }

    private fun showFaces(faces: List<Face>, width: Int, height: Int, source: String) {
        binding.faceOverlay.setFaces(faces.map { it.boundingBox }, width, height)
        val first = faces.firstOrNull()
        val status = if (faces.isEmpty()) "Không thấy khuôn mặt" else "${faces.size} khuôn mặt · yaw ${first!!.headEulerAngleY.toInt()}° · pitch ${first.headEulerAngleX.toInt()}°"
        binding.tvMlResult.text = "$source · model ${modeName()}\n$status"
        val payload = JSONObject()
            .put("source", source.lowercase())
            .put("model", modeName().lowercase())
            .put("faces", faces.size)
            .put("yaw", first?.headEulerAngleY?.toDouble() ?: JSONObject.NULL)
            .put("pitch", first?.headEulerAngleX?.toDouble() ?: JSONObject.NULL)
            .put("roll", first?.headEulerAngleZ?.toDouble() ?: JSONObject.NULL)
            .put("smile", first?.smilingProbability?.toDouble() ?: JSONObject.NULL)
            .put("session", attendanceRepository.currentSession())
            .put("presentCount", attendanceRepository.attendance().count { it.session == attendanceRepository.currentSession() })
            .put("registeredCount", attendanceRepository.students().size)
            .put("attendees", attendanceSnapshot(attendanceRepository.currentSession()))
            .put("updatedAt", System.currentTimeMillis())
            .toString()
        if (System.currentTimeMillis() > attendancePayloadUntil) {
            latestPayload = payload
            notifyWebClients()
        }
    }

    private fun checkInFromCamera() {
        val photo = binding.previewView.bitmap
        if (photo == null) {
            showMessage("Camera chưa sẵn sàng. Hãy bật camera và nhìn thẳng vào máy.")
            return
        }
        binding.btnCheckIn.isEnabled = false
        binding.tvMlResult.text = "Đang kiểm tra khuôn mặt và đối sánh danh sách sinh viên…"
        val session = binding.etSession.text.toString().trim().ifBlank { attendanceRepository.currentSession() }
        attendanceRepository.setSession(session)
        activeDetector().process(InputImage.fromBitmap(photo, 0))
            .addOnSuccessListener { faces ->
                if (faces.size != 1) {
                    photo.recycle()
                    binding.btnCheckIn.isEnabled = true
                    showMessage("Điểm danh cần đúng một khuôn mặt trong khung hình; tìm thấy ${faces.size}.")
                    return@addOnSuccessListener
                }
                val rejection = FaceQualityValidator.rejectionReason(photo, faces.first())
                if (rejection != null) {
                    photo.recycle()
                    binding.btnCheckIn.isEnabled = cameraProvider != null
                    showMessage("Ảnh chưa đạt chất lượng: $rejection. Hãy nhìn thẳng và đưa mặt gần camera hơn.")
                    return@addOnSuccessListener
                }
                cameraExecutor.execute {
                    val result = runCatching {
                        val aligned = FaceImagePreprocessor.cropAndAlign(photo, faces.first())
                        val vector = (embeddingModel ?: FaceEmbeddingModel(this).also { embeddingModel = it }).embed(aligned)
                        aligned.recycle()
                        vector
                    }
                    photo.recycle()
                    runOnUiThread {
                        binding.btnCheckIn.isEnabled = cameraProvider != null
                        result.onSuccess { embedding -> matchAndCheckIn(embedding, session) }
                            .onFailure { showMessage("Không tạo được embedding: ${it.localizedMessage}") }
                    }
                }
            }
            .addOnFailureListener { error ->
                photo.recycle()
                binding.btnCheckIn.isEnabled = true
                showMessage("ML Kit lỗi: ${error.localizedMessage}")
            }
    }

    private fun matchAndCheckIn(embedding: FloatArray, session: String) {
        val students = attendanceRepository.students()
        if (students.isEmpty()) {
            showMessage("Chưa có sinh viên. Mở Sinh viên & sổ điểm danh để đăng ký 3–4 ảnh mỗi người.")
            return
        }
        val best = students.map { it to FaceEmbeddingModel.cosine(embedding, it.embedding) }.sortedByDescending { it.second }
        val (student, score) = best.first()
        val margin = if (best.size > 1) score - best[1].second else 1f
        val threshold = attendanceRepository.matchThreshold()
        if (score < threshold || margin < 0.035f) {
            binding.tvMlResult.text = "Không xác thực · điểm khớp ${(score * 100).toInt()}% · ngưỡng ${(threshold * 100).toInt()}%"
            showMessage("Không khớp sinh viên đã đăng ký. Điểm ${(score * 100).toInt()}%.")
            return
        }
        val inserted = attendanceRepository.markPresent(student, score, session)
        binding.tvMlResult.text = if (inserted) "ĐIỂM DANH THÀNH CÔNG\n${student.name} · ${student.id} · khớp ${(score * 100).toInt()}%" else "ĐÃ ĐIỂM DANH\n${student.name} · ${student.id} · buổi $session"
        showMessage(if (inserted) "Đã điểm danh ${student.name}." else "${student.name} đã có mặt trong buổi này.")
        val present = attendanceRepository.attendance().count { it.session == session }
        latestPayload = JSONObject()
            .put("source", "attendance")
            .put("model", "mobilefacenet")
            .put("faces", 1)
            .put("yaw", 0)
            .put("pitch", 0)
            .put("attendance", if (inserted) "checked_in" else "already_present")
            .put("studentId", student.id)
            .put("studentName", student.name)
            .put("session", session)
            .put("similarity", score.toDouble())
            .put("presentCount", present)
            .put("registeredCount", students.size)
            .put("attendees", attendanceSnapshot(session))
            .put("updatedAt", System.currentTimeMillis())
            .toString()
        attendancePayloadUntil = System.currentTimeMillis() + 3500
        notifyWebClients()
    }

    private fun attendanceSnapshot(session: String) = org.json.JSONArray().apply {
        attendanceRepository.attendance().filter { it.session == session }.forEach { record ->
            put(JSONObject()
                .put("studentId", record.studentId)
                .put("studentName", record.studentName)
                .put("similarity", record.similarity.toDouble())
                .put("updatedAt", record.timestamp))
        }
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        cameraProvider = null
        binding.faceOverlay.setFaces(emptyList(), 1, 1)
        binding.tvCameraHint.visibility = android.view.View.VISIBLE
        binding.tvCameraHint.text = "Nhấn Bật camera để bắt đầu"
        binding.btnCamera.text = "Bật camera"
        binding.btnCheckIn.isEnabled = false
        binding.tvMlResult.text = "Camera đã tắt"
    }

    private fun requestBlePermissions() {
        val needed = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (checkSelfPermission(Manifest.permission.BLUETOOTH_ADVERTISE) != PackageManager.PERMISSION_GRANTED) needed += Manifest.permission.BLUETOOTH_ADVERTISE
            if (checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) needed += Manifest.permission.BLUETOOTH_CONNECT
        }
        if (needed.isEmpty()) startBleServer() else blePermissions.launch(needed.toTypedArray())
    }

    private fun startBleServer() {
        val btAdapter = adapter
        if (btAdapter == null || !btAdapter.isEnabled) {
            showMessage("Bật Bluetooth trên điện thoại rồi thử lại.")
            return
        }
        if (!btAdapter.isMultipleAdvertisementSupported) {
            showMessage("Điện thoại này không hỗ trợ BLE advertising.")
            return
        }
        try {
            gattServer = bluetoothManager.openGattServer(this, gattCallback)
            val service = android.bluetooth.BluetoothGattService(SERVICE_UUID, android.bluetooth.BluetoothGattService.SERVICE_TYPE_PRIMARY)
            val result = BluetoothGattCharacteristic(
                RESULT_UUID,
                BluetoothGattCharacteristic.PROPERTY_READ,
                BluetoothGattCharacteristic.PERMISSION_READ
            )
            result.value = latestPayload.toByteArray(Charsets.UTF_8)
            val notifier = BluetoothGattCharacteristic(
                NOTIFY_UUID,
                BluetoothGattCharacteristic.PROPERTY_NOTIFY,
                BluetoothGattCharacteristic.PERMISSION_READ
            )
            notifier.value = byteArrayOf(1)
            notifier.addDescriptor(BluetoothGattDescriptor(CCCD_UUID, BluetoothGattDescriptor.PERMISSION_READ or BluetoothGattDescriptor.PERMISSION_WRITE))
            service.addCharacteristic(result)
            service.addCharacteristic(notifier)
            if (gattServer?.addService(service) != true) {
                showMessage("Không thể đăng ký BLE GATT service.")
                stopBleServer()
                return
            }
            val data = AdvertiseData.Builder().addServiceUuid(ParcelUuid(SERVICE_UUID)).setIncludeDeviceName(false).build()
            val settings = AdvertiseSettings.Builder()
                .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY)
                .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_MEDIUM)
                .setConnectable(true)
                .build()
            btAdapter.bluetoothLeAdvertiser?.startAdvertising(settings, data, advertiseCallback)
                ?: run { showMessage("BLE advertiser không khả dụng."); stopBleServer(); return }
            advertising = true
            binding.btnBle.text = "Tắt BLE"
            binding.tvBleStatus.text = "Đang phát · chờ web dashboard kết nối"
        } catch (error: SecurityException) {
            showMessage("Thiếu quyền Bluetooth: ${error.localizedMessage}")
        } catch (error: Exception) {
            showMessage("Không khởi động được BLE: ${error.localizedMessage}")
        }
    }

    private val advertiseCallback = object : AdvertiseCallback() {
        override fun onStartSuccess(settingsInEffect: AdvertiseSettings) {
            runOnUiThread { binding.tvBleStatus.text = if (connectedDevice == null) "Đang phát · mở web dashboard để kết nối" else "Web dashboard đã kết nối" }
        }
        override fun onStartFailure(errorCode: Int) {
            runOnUiThread { binding.tvBleStatus.text = "BLE advertising lỗi: $errorCode"; showMessage("BLE advertising thất bại ($errorCode)") }
        }
    }

    private val gattCallback = object : BluetoothGattServerCallback() {
        override fun onConnectionStateChange(device: android.bluetooth.BluetoothDevice, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) connectedDevice = device
            if (newState == BluetoothProfile.STATE_DISCONNECTED) connectedDevice = null
            runOnUiThread { binding.tvBleStatus.text = if (newState == BluetoothProfile.STATE_CONNECTED) "Web dashboard đã kết nối" else "Đang phát · chờ web dashboard kết nối" }
        }
        override fun onCharacteristicReadRequest(device: android.bluetooth.BluetoothDevice, requestId: Int, offset: Int, characteristic: BluetoothGattCharacteristic) {
            if (characteristic.uuid != RESULT_UUID) return
            val bytes = latestPayload.toByteArray(Charsets.UTF_8)
            if (offset > bytes.size) gattServer?.sendResponse(device, requestId, BluetoothGatt.GATT_INVALID_OFFSET, offset, null)
            else gattServer?.sendResponse(device, requestId, BluetoothGatt.GATT_SUCCESS, offset, bytes.copyOfRange(offset, bytes.size))
        }
        override fun onDescriptorWriteRequest(device: android.bluetooth.BluetoothDevice, requestId: Int, descriptor: BluetoothGattDescriptor, preparedWrite: Boolean, responseNeeded: Boolean, offset: Int, value: ByteArray) {
            if (descriptor.uuid == CCCD_UUID) {
                descriptor.value = value
                if (responseNeeded) gattServer?.sendResponse(device, requestId, BluetoothGatt.GATT_SUCCESS, 0, value)
            } else if (responseNeeded) gattServer?.sendResponse(device, requestId, BluetoothGatt.GATT_REQUEST_NOT_SUPPORTED, 0, null)
        }
    }

    private fun notifyWebClients() {
        val server = gattServer ?: return
        val device = connectedDevice ?: return
        val service = server.getService(SERVICE_UUID) ?: return
        val resultCharacteristic = service.getCharacteristic(RESULT_UUID) ?: return
        val notifier = service.getCharacteristic(NOTIFY_UUID) ?: return
        val bytes = latestPayload.toByteArray(Charsets.UTF_8)
        resultCharacteristic.value = bytes
        val signal = byteArrayOf(1)
        notifier.value = signal
        try {
            if (Build.VERSION.SDK_INT >= 33) server.notifyCharacteristicChanged(device, notifier, false, signal)
            else @Suppress("DEPRECATION") server.notifyCharacteristicChanged(device, notifier, false)
        } catch (_: SecurityException) { }
    }

    private fun stopBleServer() {
        try { adapter?.bluetoothLeAdvertiser?.stopAdvertising(advertiseCallback) } catch (_: SecurityException) { }
        gattServer?.close()
        gattServer = null
        connectedDevice = null
        advertising = false
        binding.btnBle.text = "Bật BLE cho web dashboard"
        binding.tvBleStatus.text = "Chưa phát BLE"
    }

    private fun showMessage(message: String) = Toast.makeText(this, message, Toast.LENGTH_LONG).show()

    override fun onDestroy() {
        stopCamera()
        stopBleServer()
        detector?.close()
        embeddingModel?.close()
        cameraExecutor.shutdown()
        super.onDestroy()
    }

    companion object {
        val SERVICE_UUID: UUID = UUID.fromString("7c6f1000-8f1b-4d4b-a752-6f656d6f0001")
        val RESULT_UUID: UUID = UUID.fromString("7c6f1001-8f1b-4d4b-a752-6f656d6f0001")
        val NOTIFY_UUID: UUID = UUID.fromString("7c6f1002-8f1b-4d4b-a752-6f656d6f0001")
        private val CCCD_UUID: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
    }
}
