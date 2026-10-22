package com.leansoft.draw.mlkitforface.face.demo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.leansoft.draw.mlkitforface.databinding.ActivityFaceDemoBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

/** Thin UI coordinator for the temporary in-memory face demo. */
class FaceDemoActivity : AppCompatActivity() {
    private lateinit var binding: ActivityFaceDemoBinding
    private lateinit var worker: ExecutorService
    private lateinit var models: FaceDemoModelRunner
    private lateinit var embeddingProcessor: FaceEmbeddingProcessor
    private var livenessController: RealtimeLivenessController? = null

    private val enrollmentSources = mutableListOf<EnrollmentSource>()
    private var enrollmentVectors: List<FloatArray> = emptyList()
    private var enrollmentCentroid: FloatArray? = null
    private var selectedModel = ModelChoice.MOBILE_FACE_NET
    private var livenessStep = 0
    private val livenessPoses = mutableListOf<String>()
    private var busy = false
    private var livenessActive = false
    private var pendingCapture = CaptureTarget.ENROLLMENT
    private var lastLivenessMessageAt = 0L

    private val captureEnrollmentPhoto =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
            if (bitmap == null) return@registerForActivityResult
            if (enrollmentSources.size >= FaceDemoRules.MAX_ENROLLMENT_PHOTOS) {
                bitmap.recycle()
                showEnrollmentStatus("Đã đủ ${FaceDemoRules.MAX_ENROLLMENT_PHOTOS} ảnh.")
                return@registerForActivityResult
            }
            enrollmentSources += EnrollmentSource(bitmap = bitmap)
            clearEmbedding()
            showEnrollmentStatus("Đã có ${enrollmentSources.size}/${FaceDemoRules.MAX_ENROLLMENT_PHOTOS} ảnh. Cần tối thiểu 3 ảnh.")
            refreshControls()
        }

    private val pickEnrollmentPhotos =
        registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
            if (uris.isEmpty()) return@registerForActivityResult
            if (uris.size !in FaceDemoRules.MIN_ENROLLMENT_PHOTOS..FaceDemoRules.MAX_ENROLLMENT_PHOTOS) {
                showEnrollmentStatus("Hãy chọn từ 3 đến 4 ảnh.")
                return@registerForActivityResult
            }
            clearEnrollmentSources()
            enrollmentSources += uris.map { EnrollmentSource(uri = it) }
            clearEmbedding()
            showEnrollmentStatus("Đã chọn ${enrollmentSources.size} ảnh từ thiết bị. Bấm tạo embedding.")
            refreshControls()
        }

    private val captureProbePhoto =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
            if (bitmap != null) verifySinglePhoto(bitmap)
        }

    private val cameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (!granted) {
                if (pendingCapture == CaptureTarget.ENROLLMENT) showEnrollmentStatus("Cần quyền camera để chụp ảnh.")
                else showLivenessStatus("Cần quyền camera để chạy liveness.")
                return@registerForActivityResult
            }
            openCameraTarget(pendingCapture)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityFaceDemoBinding.inflate(layoutInflater)
        setContentView(binding.root)
        worker = Executors.newSingleThreadExecutor()
        models = FaceDemoModelRunner(this)
        embeddingProcessor = FaceEmbeddingProcessor(this, models)

        binding.btnSelectModel.setOnClickListener { chooseModel() }
        binding.btnCaptureEnrollmentPhoto.setOnClickListener { requestCamera(CaptureTarget.ENROLLMENT) }
        binding.btnPickEnrollmentPhotos.setOnClickListener { pickEnrollmentPhotos.launch("image/*") }
        binding.btnCreateEmbedding.setOnClickListener { createEmbeddingInMemory() }
        binding.btnStartLiveness.setOnClickListener { startLiveness() }
        binding.btnStopLiveness.setOnClickListener { stopLiveness("Đã dừng liveness.") }
        binding.btnCaptureProbePhoto.setOnClickListener { requestCamera(CaptureTarget.SINGLE_PHOTO) }

        binding.btnSelectModel.text = "Model: ${selectedModel.label}"
        showEnrollmentStatus("Chưa có ảnh đăng ký.")
        binding.tvEmbeddingOutput.text = "Embedding chỉ nằm trong RAM, không lưu hồ sơ."
        showLivenessStatus("Tạo embedding trước khi chạy camera liveness.")
        refreshControls()
    }

    private fun requestCamera(target: CaptureTarget) {
        pendingCapture = target
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCameraTarget(target)
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun openCameraTarget(target: CaptureTarget) {
        when (target) {
            CaptureTarget.ENROLLMENT -> captureEnrollmentPhoto.launch(null)
            CaptureTarget.LIVE_LIVENESS -> startLivenessCamera()
            CaptureTarget.SINGLE_PHOTO -> captureProbePhoto.launch(null)
        }
    }

    private fun chooseModel() {
        if (busy || livenessActive) return
        val choices = ModelChoice.values().map { it.label }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Chọn model embedding")
            .setSingleChoiceItems(choices, selectedModel.ordinal) { dialog, which ->
                val next = ModelChoice.values()[which]
                if (next != selectedModel) {
                    selectedModel = next
                    clearEmbedding()
                    binding.btnSelectModel.text = "Model: ${selectedModel.label}"
                    binding.tvEmbeddingOutput.text = "Đã chọn ${selectedModel.label}. Bấm tạo embedding lại."
                    showLivenessStatus("Chưa có embedding của model mới.")
                    refreshControls()
                }
                dialog.dismiss()
            }
            .show()
    }

    private fun createEmbeddingInMemory() {
        if (enrollmentSources.size !in FaceDemoRules.MIN_ENROLLMENT_PHOTOS..FaceDemoRules.MAX_ENROLLMENT_PHOTOS) {
            binding.tvEmbeddingOutput.text = "Cần ít nhất 3 ảnh và tối đa 4 ảnh trước."
            return
        }
        val sources = enrollmentSources.toList()
        val model = selectedModel
        busy = true
        refreshControls()
        binding.tvEmbeddingOutput.text = "Đang chạy ${model.label} trên ${sources.size} ảnh…"
        worker.execute {
            val result = runCatching { embeddingProcessor.createEmbedding(sources, model) }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                busy = false
                result.onSuccess { embedding ->
                    enrollmentVectors = embedding.vectors
                    enrollmentCentroid = embedding.centroid
                    binding.tvEmbeddingOutput.text = buildString {
                        appendLine("EMBEDDING ĐÃ TẠO TRONG RAM")
                        appendLine("Model: ${model.label}")
                        appendLine("Mẫu hợp lệ: ${embedding.vectors.size}/${sources.size}")
                        if (embedding.rejected > 0) appendLine("Ảnh bị bỏ qua: ${embedding.rejected}")
                        appendLine("Kích thước vector: ${embedding.centroid.size}-D")
                        appendLine("Không lưu mã, tên, ảnh hoặc embedding vào repository.")
                    }
                    showLivenessStatus("Sẵn sàng. Chọn camera realtime hoặc chụp một ảnh.")
                }.onFailure {
                    clearEmbedding()
                    binding.tvEmbeddingOutput.text = "TẠO EMBEDDING THẤT BẠI\n${it.localizedMessage ?: "Lỗi xử lý ảnh"}"
                }
                refreshControls()
            }
        }
    }

    private fun startLiveness() {
        if (enrollmentCentroid == null) {
            showLivenessStatus("Cần tạo embedding trước.")
            return
        }
        livenessStep = 0
        livenessPoses.clear()
        livenessActive = true
        binding.livenessCameraContainer.visibility = android.view.View.VISIBLE
        showLivenessStatus(livenessInstruction())
        refreshControls()
        requestCamera(CaptureTarget.LIVE_LIVENESS)
    }

    private fun startLivenessCamera() {
        livenessController?.close()
        livenessController = RealtimeLivenessController(
            context = this,
            lifecycleOwner = this,
            previewView = binding.livenessPreview,
            detector = models.detector(),
            analyzerExecutor = worker,
            poseForYaw = ::poseForYaw,
            onFrame = ::onLivePose,
            onError = { message -> runOnUiThread { showLivenessStatusThrottled("Camera liveness: $message") } }
        ).also { it.start() }
    }

    private fun onLivePose(pose: String) {
        runOnUiThread {
            if (!livenessActive) return@runOnUiThread
            val expected = FaceDemoRules.LIVE_POSES[livenessStep]
            if (pose == expected) {
                livenessPoses += pose
                livenessStep += 1
                if (livenessStep >= FaceDemoRules.LIVE_POSES.size) finishLiveLiveness()
                else showLivenessStatus(livenessInstruction())
            } else {
                showLivenessStatusThrottled("Đang nhìn ${pose.lowercase(Locale.US)}. ${livenessInstruction()}")
            }
        }
    }

    private fun finishLiveLiveness() {
        if (!livenessActive) return
        livenessActive = false
        val probe = binding.livenessPreview.bitmap?.copy(Bitmap.Config.ARGB_8888, false)
        stopLivenessCamera()
        if (probe == null) {
            stopLiveness("Liveness đạt nhưng không đọc được frame camera để xác thực.")
            return
        }
        busy = true
        refreshControls()
        showLivenessStatus("Liveness đạt. Đang tạo embedding frame live và xác thực…")
        val model = selectedModel
        worker.execute {
            val result = runCatching {
                embeddingProcessor.verifyLiveBitmap(probe, model, enrollmentVectors, livenessPoses)
            }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                busy = false
                showLivenessStatus(result.getOrElse { "XÁC THỰC LỖI\n${it.localizedMessage ?: "không xử lý được frame live"}" })
                refreshControls()
            }
        }
    }

    private fun verifySinglePhoto(bitmap: Bitmap) {
        if (enrollmentCentroid == null) {
            bitmap.recycle()
            showLivenessStatus("Cần tạo embedding trước.")
            return
        }
        val model = selectedModel
        busy = true
        refreshControls()
        showLivenessStatus("Đang so khớp một ảnh với embedding bằng ${model.label}…")
        worker.execute {
            val result = runCatching {
                embeddingProcessor.verifySinglePhoto(bitmap, model, enrollmentVectors)
            }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                busy = false
                showLivenessStatus(result.getOrElse { "SO KHỚP ẢNH LỖI\n${it.localizedMessage ?: "không xử lý được ảnh"}" })
                refreshControls()
            }
        }
    }

    private fun stopLiveness(message: String) {
        livenessActive = false
        livenessStep = 0
        livenessPoses.clear()
        stopLivenessCamera()
        binding.livenessCameraContainer.visibility = android.view.View.GONE
        showLivenessStatus(message)
        refreshControls()
    }

    private fun stopLivenessCamera() {
        livenessController?.close()
        livenessController = null
    }

    private fun poseForYaw(yaw: Float): String = when {
        abs(yaw) < FaceDemoRules.FRONT_YAW_DEGREES -> "FRONT"
        yaw < 0f -> "LEFT"
        else -> "RIGHT"
    }

    private fun livenessInstruction(): String =
        "Bước ${livenessStep + 1}/${FaceDemoRules.LIVE_POSES.size}: ${when (FaceDemoRules.LIVE_POSES[livenessStep]) {
            "FRONT" -> "nhìn thẳng vào camera"
            "LEFT" -> "quay mặt sang trái"
            else -> "quay mặt sang phải"
        }}"

    private fun clearEmbedding() {
        enrollmentVectors = emptyList()
        enrollmentCentroid = null
        if (livenessActive) stopLiveness("Đã đổi ảnh/model, cần tạo embedding lại.")
        else {
            livenessPoses.clear()
            livenessStep = 0
        }
    }

    private fun clearEnrollmentSources() {
        enrollmentSources.forEach { source ->
            source.bitmap?.let { if (!it.isRecycled) it.recycle() }
        }
        enrollmentSources.clear()
    }

    private fun refreshControls() {
        val hasEnoughPhotos = enrollmentSources.size in FaceDemoRules.MIN_ENROLLMENT_PHOTOS..FaceDemoRules.MAX_ENROLLMENT_PHOTOS
        binding.btnSelectModel.isEnabled = !busy && !livenessActive
        binding.btnCaptureEnrollmentPhoto.isEnabled = !busy && !livenessActive && enrollmentSources.size < FaceDemoRules.MAX_ENROLLMENT_PHOTOS
        binding.btnPickEnrollmentPhotos.isEnabled = !busy && !livenessActive
        binding.btnCreateEmbedding.isEnabled = !busy && !livenessActive && hasEnoughPhotos
        binding.btnStartLiveness.isEnabled = !busy && !livenessActive && enrollmentCentroid != null
        binding.btnStopLiveness.isEnabled = livenessActive
        binding.btnCaptureProbePhoto.isEnabled = !busy && !livenessActive && enrollmentCentroid != null
    }

    private fun showEnrollmentStatus(value: String) {
        binding.tvEnrollmentInput.text = value
    }

    private fun showLivenessStatus(value: String) {
        binding.tvLivenessOutput.text = value
    }

    private fun showLivenessStatusThrottled(value: String) {
        val now = System.currentTimeMillis()
        if (now - lastLivenessMessageAt >= 500L) {
            lastLivenessMessageAt = now
            showLivenessStatus(value)
        }
    }

    override fun onDestroy() {
        stopLivenessCamera()
        clearEnrollmentSources()
        if (::worker.isInitialized) {
            worker.execute { models.close() }
            worker.shutdown()
        } else if (::models.isInitialized) {
            models.close()
        }
        super.onDestroy()
    }

}
