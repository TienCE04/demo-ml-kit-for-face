package com.leansoft.draw.mlkitforface

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.leansoft.draw.mlkitforface.databinding.ActivityFaceDemoBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.sqrt

/** A local-only input/output screen for enrollment embeddings and face verification. */
class FaceDemoActivity : AppCompatActivity() {
    private lateinit var binding: ActivityFaceDemoBinding
    private lateinit var repository: AttendanceRepository
    private lateinit var worker: ExecutorService
    private var detector: FaceDetector? = null
    private var embedder: FaceEmbeddingModel? = null
    private var enrollmentUris: List<Uri> = emptyList()

    private val pickEnrollmentPhotos =
        registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
            if (uris.isEmpty()) return@registerForActivityResult
            if (uris.size !in 3..4) {
                binding.tvEnrollmentInput.text = "Đã chọn ${uris.size} ảnh. Hãy chọn đúng 3–4 ảnh."
                enrollmentUris = emptyList()
                return@registerForActivityResult
            }
            enrollmentUris = uris
            binding.tvEnrollmentInput.text = buildString {
                append("Đầu vào: ${uris.size} ảnh. ML Kit sẽ cần đúng 1 khuôn mặt mỗi ảnh.\n")
                append("Yaw: |yaw| < ${FRONT_YAW_DEGREES.fmt()}° là FRONT; yaw trái/phải từ ${FRONT_YAW_DEGREES.fmt()}° đến ${FaceQualityValidator.ENROLLMENT_MAX_YAW_DEGREES.fmt()}°. Pitch tối đa ${FaceQualityValidator.ENROLLMENT_MAX_PITCH_DEGREES.fmt()}°.\n")
                append("Ảnh không đạt sẽ được ghi lý do; tối thiểu 3 ảnh hợp lệ và đủ FRONT / LEFT / RIGHT.")
            }
        }

    private val pickProbePhoto =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            if (uri != null) verifyPhoto(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityFaceDemoBinding.inflate(layoutInflater)
        setContentView(binding.root)
        repository = AttendanceRepository(this)
        worker = Executors.newSingleThreadExecutor()

        binding.btnPickEnrollmentPhotos.setOnClickListener {
            pickEnrollmentPhotos.launch("image/*")
        }
        binding.btnCreateEmbedding.setOnClickListener { enrollSelectedPhotos() }
        binding.btnPickProbePhoto.setOnClickListener { pickProbePhoto.launch("image/*") }
        binding.tvEnrollmentInput.text = "Chưa chọn ảnh đăng ký."
        binding.tvEnrollmentOutput.text = modelDescription()
        binding.tvAuthOutput.text = "Chưa có ảnh xác thực."
    }

    private fun enrollSelectedPhotos() {
        val id = binding.etStudentId.text.toString().trim()
        val name = binding.etStudentName.text.toString().trim()
        if (id.isBlank() || name.isBlank()) {
            binding.tvEnrollmentOutput.text = "Đầu vào chưa hợp lệ: hãy nhập mã sinh viên và họ tên."
            return
        }
        if (!binding.cbConsent.isChecked) {
            binding.tvEnrollmentOutput.text = "Đầu vào chưa hợp lệ: cần ghi nhận đồng ý tạo embedding."
            return
        }
        if (enrollmentUris.size !in 3..4) {
            binding.tvEnrollmentOutput.text = "Đầu vào chưa hợp lệ: hãy chọn 3–4 ảnh trước."
            return
        }

        val existing = repository.students().firstOrNull { it.id.equals(id, ignoreCase = true) }
        if (existing != null) {
            AlertDialog.Builder(this)
                .setTitle("Cập nhật hồ sơ embedding?")
                .setMessage("Mã $id đã có hồ sơ. Tiếp tục sẽ thay các embedding cũ bằng bộ ảnh mới.")
                .setNegativeButton("Hủy", null)
                .setPositiveButton("Tiếp tục") { _, _ -> processEnrollment(id, name, enrollmentUris.toList()) }
                .show()
        } else {
            processEnrollment(id, name, enrollmentUris.toList())
        }
    }

    private fun processEnrollment(id: String, name: String, uris: List<Uri>) {
        setBusy(true)
        binding.tvEnrollmentOutput.text = "Đang phát hiện khuôn mặt, kiểm tra chất lượng và tạo embedding…"
        worker.execute {
            val report = runCatching {
                val outcomes = mutableListOf<EnrollmentSample>()
                uris.forEachIndexed { index, uri ->
                    outcomes += createEnrollmentSample(index + 1, uri)
                }
                val valid = outcomes.mapNotNull { sample -> sample.vector?.let { vector -> sample to vector } }
                val poseSet = valid.map { it.first.pose }.toSet()
                val requiredPoses = setOf("FRONT", "LEFT", "RIGHT")
                val registered = valid.size in 3..4 && requiredPoses.all(poseSet::contains)
                if (registered) {
                    val vectors = valid.map { it.second }
                    val types = valid.map { it.first.pose }
                    val centroid = FaceEmbeddingModel.centroid(vectors)
                    repository.saveStudent(
                        EnrolledStudent(
                            id = id,
                            name = name,
                            embedding = centroid,
                            sampleCount = vectors.size,
                            sampleTypes = types,
                            embeddingSamples = vectors,
                            consentVersion = "ATTENDANCE_FACE_VERIFICATION_V1",
                            consentAt = System.currentTimeMillis()
                        )
                    )
                }
                enrollmentReport(id, name, outcomes, registered, valid.map { it.second })
            }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                setBusy(false)
                binding.tvEnrollmentOutput.text = report.getOrElse {
                    "KHÔNG LƯU ĐƯỢC HỒ SƠ\n${it.localizedMessage ?: "lỗi khi tạo embedding hoặc lưu dữ liệu"}"
                }
            }
        }
    }

    private fun createEnrollmentSample(index: Int, uri: Uri): EnrollmentSample {
        var bitmap: Bitmap? = null
        return try {
            val source = FaceImagePreprocessor.loadCorrectly(this, uri)
            bitmap = source
            val faces = Tasks.await(faceDetector().process(InputImage.fromBitmap(source, 0)))
            if (faces.size != 1) {
                EnrollmentSample(index, error = "cần đúng 1 khuôn mặt, phát hiện ${faces.size}")
            } else {
                val face = faces.single()
                val rejection = FaceQualityValidator.rejectionReason(source, face, allowEnrollmentPose = true)
                if (rejection != null) {
                    EnrollmentSample(
                        index,
                        yaw = face.headEulerAngleY,
                        pitch = face.headEulerAngleX,
                        error = rejection
                    )
                } else {
                    val aligned = FaceImagePreprocessor.cropAndAlign(source, face)
                    val vector = try {
                        activeEmbedder().embed(aligned)
                    } finally {
                        aligned.recycle()
                    }
                    EnrollmentSample(
                        index = index,
                        pose = enrollmentPose(face),
                        yaw = face.headEulerAngleY,
                        pitch = face.headEulerAngleX,
                        vector = vector
                    )
                }
            }
        } catch (error: Exception) {
            EnrollmentSample(index, error = error.localizedMessage ?: "không xử lý được ảnh")
        } finally {
            bitmap?.recycle()
        }
    }

    private fun verifyPhoto(uri: Uri) {
        val students = repository.students()
        if (students.isEmpty()) {
            binding.tvAuthOutput.text = "Không có hồ sơ để so khớp. Hãy tạo embedding sinh viên trước."
            return
        }
        setBusy(true)
        binding.tvAuthOutput.text = "Đang chạy ML Kit → MobileFaceNet → cosine similarity…"
        val threshold = repository.matchThreshold()
        worker.execute {
            val result = runCatching {
                val bitmap = FaceImagePreprocessor.loadCorrectly(this, uri)
                try {
                    val faces = Tasks.await(faceDetector().process(InputImage.fromBitmap(bitmap, 0)))
                    require(faces.size == 1) { "Ảnh kiểm tra cần đúng 1 khuôn mặt; phát hiện ${faces.size}." }
                    val face = faces.single()
                    val rejection = FaceQualityValidator.rejectionReason(bitmap, face)
                    require(rejection == null) { "Ảnh kiểm tra không đạt: $rejection." }
                    val aligned = FaceImagePreprocessor.cropAndAlign(bitmap, face)
                    val vector = try {
                        activeEmbedder().embed(aligned)
                    } finally {
                        aligned.recycle()
                    }
                    buildVerificationReport(vector, face, students, threshold)
                } finally {
                    bitmap.recycle()
                }
            }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                setBusy(false)
                binding.tvAuthOutput.text = result.getOrElse {
                    "XÁC THỰC KHÔNG THỰC HIỆN ĐƯỢC\nĐầu vào: 1 ảnh kiểm tra.\nLý do: ${it.localizedMessage ?: "lỗi xử lý ảnh"}"
                }
            }
        }
    }

    private fun buildVerificationReport(
        probe: FloatArray,
        face: Face,
        students: List<EnrolledStudent>,
        threshold: Float
    ): String {
        val ranked = students.map { student ->
            val templates = student.embeddingSamples.ifEmpty { listOf(student.embedding) }
            val perTemplate = templates.mapIndexed { index, vector ->
                val pose = student.sampleTypes.getOrNull(index) ?: "MẪU ${index + 1}"
                pose to FaceEmbeddingModel.cosine(probe, vector)
            }
            val best = perTemplate.maxByOrNull { it.second } ?: ("CENTROID" to FaceEmbeddingModel.cosine(probe, student.embedding))
            CandidateScore(
                student = student,
                bestPose = best.first,
                bestScore = best.second,
                centroidScore = FaceEmbeddingModel.cosine(probe, student.embedding),
                templateScores = perTemplate
            )
        }.sortedByDescending { it.bestScore }

        val best = ranked.first()
        val margin = if (ranked.size > 1) best.bestScore - ranked[1].bestScore else 1f
        val accepted = best.bestScore >= threshold && margin >= DemoAttendanceGateway.MIN_MARGIN
        return buildString {
            appendLine(if (accepted) "KẾT QUẢ: MATCH" else "KẾT QUẢ: NO MATCH")
            appendLine("Đầu vào: 1 ảnh · 1 khuôn mặt · yaw ${face.headEulerAngleY.fmt()}° · pitch ${face.headEulerAngleX.fmt()}°.")
            appendLine("MobileFaceNet: vector ${probe.size}-D, L2 norm ${l2Norm(probe).fmt()}.")
            appendLine("Điều kiện cosine: ${best.bestScore.fmt()} ≥ ${threshold.fmt()} · ${if (best.bestScore >= threshold) "PASS" else "FAIL"}.")
            appendLine("Điều kiện margin: ${margin.fmt()} ≥ ${DemoAttendanceGateway.MIN_MARGIN.fmt()} · ${if (margin >= DemoAttendanceGateway.MIN_MARGIN) "PASS" else "FAIL"}.")
            appendLine("Kết quả chỉ là kiểm tra xác thực cục bộ; không tạo lượt điểm danh.")
            appendLine()
            appendLine("Xếp hạng hồ sơ (điểm hồ sơ = cosine cao nhất với từng embedding mẫu):")
            ranked.forEachIndexed { index, candidate ->
                appendLine("${index + 1}. ${candidate.student.name} (${candidate.student.id}) · best ${candidate.bestScore.fmt()} · mẫu ${candidate.bestPose} · centroid ${candidate.centroidScore.fmt()}")
                appendLine("   Mẫu: ${candidate.templateScores.joinToString { (pose, score) -> "$pose=${score.fmt()}" }}")
            }
            appendLine()
            appendLine("Embedding ảnh kiểm tra đủ ${probe.size} giá trị:")
            appendLine(vectorValues(probe))
        }
    }

    private fun enrollmentReport(
        id: String,
        name: String,
        outcomes: List<EnrollmentSample>,
        registered: Boolean,
        vectors: List<FloatArray>
    ): String = buildString {
        appendLine(if (registered) "ĐĂNG KÝ EMBEDDING THÀNH CÔNG" else "CHƯA ĐĂNG KÝ ĐƯỢC")
        appendLine("Hồ sơ: $name ($id) · ${outcomes.size} ảnh đầu vào.")
        appendLine("Pipeline: ảnh → ML Kit (1 mặt, góc, độ nét) → crop/align → MobileFaceNet 112×112 RGB → embedding L2-normalized.")
        appendLine("Điều kiện đăng ký: tối thiểu 3 ảnh đạt và có đủ FRONT / LEFT / RIGHT; ảnh thứ 4 là mẫu bổ sung.")
        appendLine("Phân tư thế theo yaw ML Kit: |yaw| < ${FRONT_YAW_DEGREES.fmt()}° là FRONT; yaw trái/phải từ ${FRONT_YAW_DEGREES.fmt()}° đến ${FaceQualityValidator.ENROLLMENT_MAX_YAW_DEGREES.fmt()}°; pitch tối đa ${FaceQualityValidator.ENROLLMENT_MAX_PITCH_DEGREES.fmt()}°.")
        appendLine()
        outcomes.forEach { sample ->
            val vector = sample.vector
            if (vector == null) {
                appendLine("Ảnh ${sample.index}: REJECT · ${sample.error ?: "không tạo được embedding"}${sample.yaw?.let { " · yaw ${it.fmt()}°" } ?: ""}${sample.pitch?.let { " · pitch ${it.fmt()}°" } ?: ""}")
            } else {
                appendLine("Ảnh ${sample.index}: ACCEPT · ${sample.pose} · yaw ${sample.yaw?.fmt() ?: "?"}° · pitch ${sample.pitch?.fmt() ?: "?"}° · ${vector.size}-D · norm ${l2Norm(vector).fmt()}")
                appendLine("   Embedding đủ ${vector.size} giá trị:")
                appendLine(vectorValues(vector))
            }
        }
        appendLine()
        if (registered) {
            val centroid = FaceEmbeddingModel.centroid(vectors)
            appendLine("Lưu trữ: ${vectors.size} vector mẫu + centroid ${centroid.size}-D trong kho cục bộ mã hóa.")
            appendLine("Centroid đủ ${centroid.size} giá trị:")
            appendLine(vectorValues(centroid))
            appendLine("Ảnh gốc không được lưu. Xác thực sẽ so ảnh mới với từng vector mẫu và lấy cosine cao nhất theo hồ sơ.")
        } else {
            val acceptedPoses = outcomes.filter { it.vector != null }.joinToString { it.pose }
            val missing = setOf("FRONT", "LEFT", "RIGHT") - outcomes.filter { it.vector != null }.map { it.pose }.toSet()
            appendLine("Chưa lưu hồ sơ. Ảnh đạt: ${outcomes.count { it.vector != null }}/${outcomes.size}; tư thế đạt: ${acceptedPoses.ifBlank { "chưa có" }}.")
            if (missing.isNotEmpty()) appendLine("Cần bổ sung tư thế: ${missing.joinToString()}.")
            if (outcomes.count { it.vector != null } < 3) appendLine("Cần ít nhất 3 ảnh vượt kiểm tra chất lượng.")
        }
    }

    private fun modelDescription(): String = buildString {
        appendLine("Model đầu ra: MobileFaceNet có sẵn trong assets/mobile_face_net.tflite.")
        appendLine("Input tensor: [1,112,112,3], RGB chuẩn hóa [-1,1]. Output: [1,192], embedding đã L2-normalize.")
        appendLine("Chưa có hồ sơ nào được tạo từ màn hình này.")
    }

    private fun faceDetector(): FaceDetector {
        detector?.let { return it }
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setMinFaceSize(0.12f)
            .build()
        return FaceDetection.getClient(options).also { detector = it }
    }

    private fun activeEmbedder(): FaceEmbeddingModel =
        embedder ?: FaceEmbeddingModel(this).also { embedder = it }

    private fun enrollmentPose(face: Face): String = when {
        kotlin.math.abs(face.headEulerAngleY) < FRONT_YAW_DEGREES -> "FRONT"
        face.headEulerAngleY < 0f -> "LEFT"
        else -> "RIGHT"
    }

    private fun setBusy(busy: Boolean) {
        binding.btnPickEnrollmentPhotos.isEnabled = !busy
        binding.btnCreateEmbedding.isEnabled = !busy
        binding.btnPickProbePhoto.isEnabled = !busy
    }

    private fun Float.fmt(): String = String.format(Locale.US, "%.4f", this)

    private fun vectorValues(vector: FloatArray): String =
        vector.toList().chunked(VECTOR_VALUES_PER_LINE).joinToString("\n      ") { row ->
            row.joinToString(", ") { it.fmt() }
        }

    private fun l2Norm(vector: FloatArray): Float =
        sqrt(vector.sumOf { it.toDouble() * it.toDouble() }).toFloat()

    override fun onDestroy() {
        if (::worker.isInitialized) {
            worker.execute {
                detector?.close()
                detector = null
                embedder?.close()
                embedder = null
            }
            worker.shutdown()
        } else {
            detector?.close()
            embedder?.close()
        }
        super.onDestroy()
    }

    private data class EnrollmentSample(
        val index: Int,
        val pose: String = "UNKNOWN",
        val yaw: Float? = null,
        val pitch: Float? = null,
        val vector: FloatArray? = null,
        val error: String? = null
    )

    private data class CandidateScore(
        val student: EnrolledStudent,
        val bestPose: String,
        val bestScore: Float,
        val centroidScore: Float,
        val templateScores: List<Pair<String, Float>>
    )

    companion object {
        private const val FRONT_YAW_DEGREES = 12f
        private const val VECTOR_VALUES_PER_LINE = 12
    }
}
