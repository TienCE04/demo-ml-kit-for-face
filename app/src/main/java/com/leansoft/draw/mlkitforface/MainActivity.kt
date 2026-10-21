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
import android.bluetooth.le.AdvertisingSet
import android.bluetooth.le.AdvertisingSetCallback
import android.bluetooth.le.AdvertisingSetParameters
import android.bluetooth.le.BluetoothLeScanner
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Rect
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
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
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private val processing = AtomicBoolean(false)
    private var detector: FaceDetector? = null
    private var embeddingModel: FaceEmbeddingModel? = null
    private lateinit var attendanceRepository: AttendanceRepository
    private lateinit var authGateway: DemoAttendanceGateway
    private var cameraProvider: ProcessCameraProvider? = null
    private var gattServer: BluetoothGattServer? = null
    private var advertising = false
    private var advertisingSet: AdvertisingSet? = null
    private var currentBroadcast: DemoAttendanceGateway.Broadcast? = null
    private var connectedDevice: android.bluetooth.BluetoothDevice? = null
    private var bleScanner: BluetoothLeScanner? = null
    private var scanning = false
    private var scanStartedAt = 0L
    private val scanSamples = mutableListOf<Int>()
    private var activeAttempt: DemoAttendanceGateway.Attempt? = null
    private var faceChallenge: DemoAttendanceGateway.FaceChallenge? = null
    private val rotationHandler = Handler(Looper.getMainLooper())
    private val rotateTask = object : Runnable {
        override fun run() {
            rotateBleToken()
            if (advertising) rotationHandler.postDelayed(this, DemoAttendanceGateway.DEFAULT_ROTATE_MS)
        }
    }
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
        authGateway = DemoAttendanceGateway(attendanceRepository)
        binding.etSession.setText(attendanceRepository.currentSession())
        latestPayload = JSONObject()
            .put("source", "camera")
            .put("model", "fast")
            .put("faces", 0)
            .put("session", attendanceRepository.currentSession())
            .put("presentCount", attendanceRepository.attendance().count { it.session == attendanceRepository.currentSession() && it.status in setOf("PRESENT", "LATE") })
            .put("registeredCount", attendanceRepository.students().size)
            .put("attendees", attendanceSnapshot(attendanceRepository.currentSession()))
            .put("auth", authGateway.payloadFor(null))
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
        binding.btnFaceAuthDemo.setOnClickListener { startActivity(Intent(this, FaceDemoActivity::class.java)) }
        binding.btnCheckIn.setOnClickListener { checkInFromCamera() }
        binding.btnBleScan.setOnClickListener { beginBleVerification() }
        binding.btnEndSession.setOnClickListener { endAttendanceSession() }
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
            .put("presentCount", attendanceRepository.attendance().count { it.session == attendanceRepository.currentSession() && it.status in setOf("PRESENT", "LATE") })
            .put("registeredCount", attendanceRepository.students().size)
            .put("attendees", attendanceSnapshot(attendanceRepository.currentSession()))
            .put("auth", authGateway.payloadFor(activeAttempt))
            .put("updatedAt", System.currentTimeMillis())
            .toString()
        if (System.currentTimeMillis() > attendancePayloadUntil) {
            latestPayload = payload
            notifyWebClients()
        }
    }

    private fun checkInFromCamera() {
        if (!isFaceStepReady()) {
            beginBleVerification()
            return
        }
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
                        result.onSuccess { embedding -> submitFaceEvidence(embedding, session) }
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

    private fun submitFaceEvidence(embedding: FloatArray, session: String) {
        val attempt = activeAttempt
        val challenge = faceChallenge
        if (attempt == null || challenge == null) {
            showMessage("Chưa có face challenge. Hãy xác minh BLE trước.")
            return
        }
        val result = authGateway.verifyFaceEvidence(
            attemptId = attempt.id,
            embedding = embedding,
            challengeNonce = challenge.nonce,
            livenessPassed = true,
            sessionCode = session
        )
        activeAttempt = result.attempt ?: attempt
        binding.tvAuthStatus.text = "Xác thực: ${result.code} · server similarity ${(result.similarity * 100).roundToInt()}% · margin ${(result.margin * 100).roundToInt()}%"
        if (!result.accepted || result.student == null) {
            binding.tvMlResult.text = "KHÔNG XÁC THỰC\n${result.message}"
            showMessage("${result.message} (${result.code})")
            publishAuthPayload(session, result)
            return
        }
        val student = result.student
        val inserted = result.attendance == "checked_in"
        binding.tvMlResult.text = if (inserted) {
            "ĐIỂM DANH THÀNH CÔNG\n${student.name} · ${student.id} · server khớp ${(result.similarity * 100).roundToInt()}%"
        } else {
            "ĐÃ ĐIỂM DANH\n${student.name} · ${student.id} · buổi $session"
        }
        showMessage(if (inserted) "Đã điểm danh ${student.name}." else "${student.name} đã có mặt trong buổi này.")
        publishAuthPayload(session, result)
    }

    private fun endAttendanceSession() {
        val session = binding.etSession.text.toString().trim().ifBlank { attendanceRepository.currentSession() }
        val absentCount = attendanceRepository.finalizeSession(session)
        authGateway.stopSession(session)
        stopBleServer()
        binding.tvAuthStatus.text = "Phiên đã ENDED · gateway chốt $absentCount sinh viên ABSENT"
        binding.tvMlResult.text = "PHIÊN ĐÃ KẾT THÚC\nCác evidence sau thời điểm này không được chấp nhận."
        publishAuthPayload(session)
        showMessage("Đã kết thúc phiên và chốt $absentCount sinh viên vắng.")
    }

    private fun isFaceStepReady(): Boolean =
        activeAttempt?.status == "VERIFYING"
            && faceChallenge?.used == false
            && (faceChallenge?.expiresAt ?: 0L) > System.currentTimeMillis()

    private fun publishAuthPayload(session: String, result: DemoAttendanceGateway.FaceResult? = null) {
        val students = attendanceRepository.students()
        val payload = JSONObject()
            .put("source", if (result == null) "ble" else "attendance")
            .put("model", "mobilefacenet")
            .put("faces", if (result == null) 0 else 1)
            .put("session", session)
            .put("presentCount", attendanceRepository.attendance().count { it.session == session && it.status in setOf("PRESENT", "LATE") })
            .put("registeredCount", students.size)
            .put("attendees", attendanceSnapshot(session))
            .put("auth", authGateway.payloadFor(activeAttempt, result))
            .put("updatedAt", System.currentTimeMillis())
        result?.student?.let { student ->
            payload.put("attendance", result.attendance)
                .put("studentId", student.id)
                .put("studentName", student.name)
                .put("similarity", result.similarity.toDouble())
        }
        latestPayload = payload.toString()
        attendancePayloadUntil = System.currentTimeMillis() + 3500
        notifyWebClients()
    }

    private fun attendanceSnapshot(session: String) = org.json.JSONArray().apply {
        attendanceRepository.attendance().filter { it.session == session }.forEach { record ->
            put(JSONObject()
                .put("studentId", record.studentId)
                .put("studentName", record.studentName)
                .put("similarity", record.similarity.toDouble())
                .put("status", record.status)
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

    /** Start the student side of the protocol. If this phone is also the local
     * broadcaster, use the same bytes as a deterministic demo fallback. */
    private fun beginBleVerification() {
        val session = binding.etSession.text.toString().trim().ifBlank { attendanceRepository.currentSession() }
        attendanceRepository.setSession(session)
        if (isFaceStepReady()) {
            binding.tvAuthStatus.text = "Xác thực: BLE đã VERIFIED · có thể gửi face evidence"
            return
        }
        val broadcast = currentBroadcast
        if (broadcast != null && broadcast.sessionCode == session) {
            val attempt = authGateway.createAttempt(session, UUID.randomUUID().toString())
            activeAttempt = attempt
            val now = System.currentTimeMillis()
            val result = authGateway.verifyBleEvidence(
                attempt.id,
                DemoAttendanceGateway.BleEvidence(
                    sessionCode = session,
                    token = broadcast.token,
                    scanCount = DemoAttendanceGateway.MIN_SCAN_COUNT,
                    scanDurationMs = DemoAttendanceGateway.MIN_SCAN_DURATION_MS,
                    rssiMin = -52,
                    rssiMax = -44,
                    rssiAverage = -48,
                    firstSeenAt = now - DemoAttendanceGateway.MIN_SCAN_DURATION_MS,
                    lastSeenAt = now
                )
            )
            applyBleResult(result, session)
            return
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED
        ) {
            requestBlePermissions()
            return
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED
        ) {
            requestBlePermissions()
            return
        }
        val scanner = adapter?.bluetoothLeScanner
        if (scanner == null) {
            showMessage("BLE scanner không khả dụng trên thiết bị này.")
            return
        }
        val attempt = authGateway.createAttempt(session, UUID.randomUUID().toString())
        activeAttempt = attempt
        scanSamples.clear()
        scanStartedAt = System.currentTimeMillis()
        scanning = true
        binding.btnBleScan.text = "Đang quét BLE…"
        binding.tvAuthStatus.text = "Xác thực: SCAN_BLE · giữ điện thoại gần máy giảng viên"
        bleScanner = scanner
        try {
            scanner.startScan(
                listOf(android.bluetooth.le.ScanFilter.Builder().setServiceUuid(ParcelUuid(SERVICE_UUID)).build()),
                android.bluetooth.le.ScanSettings.Builder()
                    .setScanMode(android.bluetooth.le.ScanSettings.SCAN_MODE_LOW_LATENCY)
                    .build(),
                scanCallback
            )
        } catch (error: SecurityException) {
            scanning = false
            showMessage("Thiếu quyền quét Bluetooth: ${error.localizedMessage}")
        }
    }

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val bytes = result.scanRecord?.getServiceData(ParcelUuid(SERVICE_UUID)) ?: return
            val payload = bytes.toString(Charsets.UTF_8)
            val parts = payload.split('|', limit = 3)
            if (parts.size != 3 || parts[0] != "v1") return
            val session = binding.etSession.text.toString().trim().ifBlank { attendanceRepository.currentSession() }
            if (parts[1] != session) return
            if (!scanning) return
            scanSamples += result.rssi
            val elapsed = System.currentTimeMillis() - scanStartedAt
            binding.tvAuthStatus.text = "Xác thực: SCAN_BLE · ${scanSamples.size}/${DemoAttendanceGateway.MIN_SCAN_COUNT} mẫu · RSSI ${result.rssi} dBm"
            if (scanSamples.size >= DemoAttendanceGateway.MIN_SCAN_COUNT && elapsed >= DemoAttendanceGateway.MIN_SCAN_DURATION_MS) {
                stopBleScan()
                val attempt = activeAttempt ?: return
                val evidence = DemoAttendanceGateway.BleEvidence(
                    sessionCode = session,
                    token = parts[2],
                    scanCount = scanSamples.size,
                    scanDurationMs = elapsed,
                    rssiMin = scanSamples.minOrNull() ?: result.rssi,
                    rssiMax = scanSamples.maxOrNull() ?: result.rssi,
                    rssiAverage = scanSamples.average().roundToInt(),
                    firstSeenAt = scanStartedAt,
                    lastSeenAt = System.currentTimeMillis()
                )
                applyBleResult(authGateway.verifyBleEvidence(attempt.id, evidence), session)
            }
        }

        override fun onScanFailed(errorCode: Int) {
            runOnUiThread {
                stopBleScan()
                binding.tvAuthStatus.text = "Xác thực: quét BLE lỗi $errorCode"
                showMessage("BLE scan thất bại ($errorCode).")
            }
        }
    }

    private fun stopBleScan() {
        if (!scanning) return
        scanning = false
        try { bleScanner?.stopScan(scanCallback) } catch (_: SecurityException) { }
        bleScanner = null
        binding.btnBleScan.text = "Quét BLE để xác minh phiên"
    }

    private fun applyBleResult(result: DemoAttendanceGateway.BleResult, session: String) {
        activeAttempt = result.attempt ?: activeAttempt
        faceChallenge = activeAttempt?.challenge
        if (result.accepted && activeAttempt?.status == "VERIFYING") {
            binding.tvAuthStatus.text = "Xác thực: BLE VERIFIED · challenge ${faceChallenge?.type} · mở camera để gửi face evidence"
            binding.btnCheckIn.isEnabled = cameraProvider != null
            publishAuthPayload(session)
            showMessage("BLE hợp lệ. Bây giờ mới thực hiện xác minh khuôn mặt.")
        } else {
            binding.tvAuthStatus.text = "Xác thực BLE bị từ chối · ${result.code}"
            showMessage(result.message)
            publishAuthPayload(session)
        }
    }

    private fun requestBlePermissions() {
        val needed = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (checkSelfPermission(Manifest.permission.BLUETOOTH_ADVERTISE) != PackageManager.PERMISSION_GRANTED) needed += Manifest.permission.BLUETOOTH_ADVERTISE
            if (checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) needed += Manifest.permission.BLUETOOTH_CONNECT
            if (checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED) needed += Manifest.permission.BLUETOOTH_SCAN
        } else if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            needed += Manifest.permission.ACCESS_FINE_LOCATION
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
            val session = binding.etSession.text.toString().trim().ifBlank { attendanceRepository.currentSession() }
            attendanceRepository.setSession(session)
            currentBroadcast = authGateway.startSession(session)
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
            val advertiser = btAdapter.bluetoothLeAdvertiser
                ?: run { showMessage("BLE advertiser không khả dụng."); stopBleServer(); return }
            val parameters = AdvertisingSetParameters.Builder()
                .setLegacyMode(false)
                .setConnectable(true)
                .setScannable(false)
                .setPrimaryPhy(android.bluetooth.BluetoothDevice.PHY_LE_1M)
                .setSecondaryPhy(android.bluetooth.BluetoothDevice.PHY_LE_1M)
                .setInterval(AdvertisingSetParameters.INTERVAL_HIGH)
                .setTxPowerLevel(AdvertisingSetParameters.TX_POWER_MEDIUM)
                .build()
            advertiser.startAdvertisingSet(
                parameters,
                advertisingData(currentBroadcast!!),
                null,
                null,
                null,
                advertisingSetCallback
            )
            advertising = true
            rotationHandler.removeCallbacks(rotateTask)
            rotationHandler.postDelayed(rotateTask, DemoAttendanceGateway.DEFAULT_ROTATE_MS)
            binding.btnBle.text = "Tắt BLE"
            binding.tvBleStatus.text = "Đang phát BLE · token xoay mỗi 30 giây · chờ dashboard"
            binding.tvAuthStatus.text = "Xác thực: phiên ${session} đang phát challenge BLE"
        } catch (error: SecurityException) {
            showMessage("Thiếu quyền Bluetooth: ${error.localizedMessage}")
            stopBleServer()
        } catch (error: Exception) {
            showMessage("Không khởi động được BLE: ${error.localizedMessage}")
            stopBleServer()
        }
    }

    private fun advertisingData(broadcast: DemoAttendanceGateway.Broadcast): AdvertiseData =
        AdvertiseData.Builder()
            .setIncludeTxPowerLevel(false)
            .addServiceUuid(ParcelUuid(SERVICE_UUID))
            .addServiceData(
                ParcelUuid(SERVICE_UUID),
                "v1|${broadcast.sessionCode}|${broadcast.token}".toByteArray(Charsets.UTF_8)
            )
            .build()

    private fun rotateBleToken() {
        val session = binding.etSession.text.toString().trim().ifBlank { attendanceRepository.currentSession() }
        val rotated = authGateway.rotateToken(session) ?: return
        currentBroadcast = rotated
        try {
            advertisingSet?.setAdvertisingData(advertisingData(rotated))
            binding.tvBleStatus.text = "Đang phát BLE · token vừa xoay · hết hạn sau 90 giây"
        } catch (error: SecurityException) {
            showMessage("Không thể xoay token BLE: ${error.localizedMessage}")
        }
    }

    private val advertisingSetCallback = object : AdvertisingSetCallback() {
        override fun onAdvertisingSetStarted(advertisingSet: AdvertisingSet?, txPower: Int, status: Int) {
            runOnUiThread {
                if (status == AdvertisingSetCallback.ADVERTISE_SUCCESS) {
                    this@MainActivity.advertisingSet = advertisingSet
                    binding.tvBleStatus.text = if (connectedDevice == null) {
                        "Đang phát BLE · mở web dashboard để kết nối"
                    } else {
                        "Web dashboard đã kết nối · BLE đang phát"
                    }
                } else {
                    advertising = false
                    binding.tvBleStatus.text = "BLE advertising lỗi: $status"
                    showMessage("BLE advertising thất bại ($status)")
                }
            }
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
        stopBleScan()
        rotationHandler.removeCallbacks(rotateTask)
        try { adapter?.bluetoothLeAdvertiser?.stopAdvertisingSet(advertisingSetCallback) } catch (_: SecurityException) { }
        gattServer?.close()
        gattServer = null
        authGateway.stopSession(binding.etSession.text.toString().trim().ifBlank { attendanceRepository.currentSession() })
        advertisingSet = null
        currentBroadcast = null
        connectedDevice = null
        advertising = false
        binding.btnBle.text = "Bật BLE cho web dashboard"
        binding.tvBleStatus.text = "Chưa phát BLE"
        binding.tvAuthStatus.text = "Xác thực: chưa bắt đầu · bước 1 là BLE evidence"
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
        // Project3 contract: service data carries v1|<sessionCode>|<token>.
        val SERVICE_UUID: UUID = UUID.fromString("7b7e2401-8d9c-4ea7-a5e1-2c89b47ef201")
        val RESULT_UUID: UUID = UUID.fromString("7c6f1001-8f1b-4d4b-a752-6f656d6f0001")
        val NOTIFY_UUID: UUID = UUID.fromString("7c6f1002-8f1b-4d4b-a752-6f656d6f0001")
        private val CCCD_UUID: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
    }
}
