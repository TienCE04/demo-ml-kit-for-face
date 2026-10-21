package com.leansoft.draw.mlkitforface

import android.app.AlertDialog
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.leansoft.draw.mlkitforface.databinding.ActivityAttendanceBinding
import com.leansoft.draw.mlkitforface.databinding.DialogEnrollStudentBinding
import com.leansoft.draw.mlkitforface.databinding.ItemAttendanceRecordBinding
import com.leansoft.draw.mlkitforface.databinding.ItemEnrolledStudentBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.OutputStreamWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class AttendanceActivity : AppCompatActivity() {
    private lateinit var binding: ActivityAttendanceBinding
    private lateinit var repository: AttendanceRepository
    private lateinit var executor: ExecutorService
    private val studentsDetector: FaceDetector by lazy {
        FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL).build()
        )
    }
    private var embedder: FaceEmbeddingModel? = null
    private var pendingStudent: Pair<String, String>? = null
    private var pendingUris: List<Uri> = emptyList()
    private val pickPhotos =
        registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
            if (uris.isEmpty()) return@registerForActivityResult
            if (uris.size !in 3..4) {
                toast("Chọn từ 3 đến 4 ảnh rõ mặt của cùng một sinh viên.")
                return@registerForActivityResult
            }
            pendingUris = uris
            processEnrollmentPhoto(0, emptyList(), emptyList())
        }
    private val exportCsv =
        registerForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { uri ->
            if (uri != null) exportAttendance(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAttendanceBinding.inflate(layoutInflater)
        setContentView(binding.root)
        repository = AttendanceRepository(this)
        executor = Executors.newSingleThreadExecutor()
        bindActions()
        refresh()
    }

    private fun bindActions() {
        binding.etSession.setText(repository.currentSession())
        binding.btnSaveSession.setOnClickListener {
            repository.setSession(binding.etSession.text.toString())
            binding.etSession.setText(repository.currentSession())
            refresh()
        }

        binding.btnEnroll.setOnClickListener { showEnrollDialog() }
        binding.etThreshold.setText("%.2f".format(Locale.US, repository.matchThreshold()))
        binding.btnSaveThreshold.setOnClickListener {
            val value = binding.etThreshold.text.toString().trim().replace(',', '.').toFloatOrNull()
            if (value == null || value !in 0.1f..0.99f) toast("Nhập ngưỡng từ 0,10 đến 0,99.")
            else {
                repository.setMatchThreshold(value)
                binding.etThreshold.setText("%.2f".format(Locale.US, value))
                toast(
                    "Đã lưu ngưỡng ${
                        "%.2f".format(
                            Locale.US,
                            value
                        )
                    }. Hãy hiệu chỉnh bằng bộ ảnh kiểm tra riêng."
                )
            }
        }
        binding.btnExportCsv.setOnClickListener {
            exportCsv.launch("attendance-${repository.currentSession()}.csv")
        }
        binding.btnClearSession.setOnClickListener { confirmClearSession() }
    }

    private fun refresh() {
        val students = repository.students()
        binding.rosterContainer.removeViews(1, (binding.rosterContainer.childCount - 1).coerceAtLeast(0))
        binding.tvRosterEmpty.visibility = if (students.isEmpty()) android.view.View.VISIBLE else android.view.View.GONE
        students.forEach { student ->
            val row = ItemEnrolledStudentBinding.inflate(layoutInflater, binding.rosterContainer, false)
            val poseSummary = student.sampleTypes.distinct().joinToString("/").ifBlank { "legacy" }
            row.tvStudentInfo.text = "${student.name}\n${student.id} · ${student.sampleCount} ảnh · $poseSummary · ${student.embedding.size}-D embedding"
            row.btnRemoveStudent.setOnClickListener { confirmRemove(student.id, student.name) }
            binding.rosterContainer.addView(row.root)
        }

        val session = repository.currentSession()
        val records = repository.attendance().filter { it.session == session }
        val presentCount = records.count { it.status in setOf("PRESENT", "LATE") }
        binding.tvRosterHeader.text = "SINH VIÊN ĐÃ ĐĂNG KÝ · ${students.size}"
        binding.tvRecordsHeader.text = "ĐIỂM DANH · $session"
        binding.tvRecordsSummary.text = "Có mặt ${presentCount}/${students.size} sinh viên · vắng ${records.count { it.status == "ABSENT" }}"
        binding.tvRecordsEmpty.visibility = if (records.isEmpty()) android.view.View.VISIBLE else android.view.View.GONE
        binding.recordsContainer.removeViews(2, (binding.recordsContainer.childCount - 2).coerceAtLeast(0))
        records.forEach { entry ->
            val time =
                SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(entry.timestamp))
            val row = ItemAttendanceRecordBinding.inflate(layoutInflater, binding.recordsContainer, false)
            row.tvRecord.text = "$time  ·  ${entry.studentName} (${entry.studentId})  ·  ${entry.status}  ·  khớp ${(entry.similarity * 100).toInt()}%"
            binding.recordsContainer.addView(row.root)
        }
    }

    private fun showEnrollDialog() {
        val dialogBinding = DialogEnrollStudentBinding.inflate(layoutInflater)
        AlertDialog.Builder(this).setTitle("Đăng ký sinh viên").setView(dialogBinding.root)
            .setNegativeButton("Hủy", null)
            .setPositiveButton("Chọn ảnh") { _, _ ->
                val studentId = dialogBinding.etStudentId.text.toString().trim()
                val studentName = dialogBinding.etStudentName.text.toString().trim()
                if (!dialogBinding.cbConsent.isChecked) toast("Cần ghi nhận consent trước khi tạo embedding.")
                else if (studentId.isBlank() || studentName.isBlank()) toast("Nhập mã sinh viên và họ tên trước.")
                else {
                    pendingStudent = studentId to studentName
                    pickPhotos.launch("image/*")
                }
            }.show()
    }

    private fun processEnrollmentPhoto(index: Int, vectors: List<FloatArray>, sampleTypes: List<String>) {
        if (index >= pendingUris.size) {
            val student = pendingStudent ?: return
            val required = setOf("FRONT", "LEFT", "RIGHT")
            if (vectors.size !in 3..4 || !required.all(sampleTypes::contains)) {
                pendingStudent = null
                setTitle("Sinh viên & sổ điểm danh")
                toast("Bộ ảnh cần đủ 3 tư thế: chính diện, nghiêng trái và nghiêng phải (tối đa thêm 1 ảnh).")
                return
            }
            val centroid = FaceEmbeddingModel.centroid(vectors)
            repository.saveStudent(
                EnrolledStudent(
                    student.first,
                    student.second,
                    centroid,
                    vectors.size,
                    sampleTypes,
                    vectors,
                    "ATTENDANCE_FACE_VERIFICATION_V1",
                    System.currentTimeMillis()
                )
            )
            pendingStudent = null
            refresh()
            toast("${student.second} đã đăng ký bằng ${vectors.size} ảnh.")
            return
        }
        runOnUiThread { setTitle("Đang xử lý ảnh ${index + 1}/${pendingUris.size}…") }
        executor.execute {
            val bitmap = runCatching {
                FaceImagePreprocessor.loadCorrectly(
                    this,
                    pendingUris[index]
                )
            }.getOrElse {
                runOnUiThread { setTitle("Sinh viên & sổ điểm danh"); toast("Không đọc được ảnh ${index + 1}: ${it.localizedMessage}") }
                return@execute
            }
            studentsDetector.process(InputImage.fromBitmap(bitmap, 0))
                .addOnSuccessListener(ContextCompat.getMainExecutor(this)) { faces ->
                    if (faces.size != 1) {
                        bitmap.recycle()
                        setTitle("Sinh viên & sổ điểm danh")
                        toast("Ảnh ${index + 1}: cần đúng một khuôn mặt, tìm thấy ${faces.size}.")
                        return@addOnSuccessListener
                    }
                    val rejection = FaceQualityValidator.rejectionReason(bitmap, faces.first(), allowEnrollmentPose = true)
                    if (rejection != null) {
                        bitmap.recycle()
                        setTitle("Sinh viên & sổ điểm danh")
                        toast("Ảnh ${index + 1} không hợp lệ: $rejection. Chọn lại bộ ảnh có ít nhất 3 ảnh đạt.")
                        return@addOnSuccessListener
                    }
                    executor.execute {
                        val vector = runCatching {
                            (embedder ?: FaceEmbeddingModel(this).also {
                                embedder = it
                            }).embed(FaceImagePreprocessor.cropAndAlign(bitmap, faces.first()))
                        }
                        bitmap.recycle()
                        vector.onSuccess { next ->
                            val pose = enrollmentPose(faces.first())
                            runOnUiThread {
                                processEnrollmentPhoto(
                                    index + 1,
                                    vectors + next,
                                    sampleTypes + pose
                                )
                            }
                        }
                            .onFailure {
                                runOnUiThread {
                                    setTitle("Sinh viên & sổ điểm danh"); toast(
                                    "Embedding lỗi: ${it.localizedMessage}"
                                )
                                }
                            }
                    }
                }
                .addOnFailureListener(ContextCompat.getMainExecutor(this)) { error ->
                    bitmap.recycle()
                    setTitle("Sinh viên & sổ điểm danh")
                    toast("ML Kit lỗi: ${error.localizedMessage}")
                }
        }
    }

    private fun confirmClearSession() {
        AlertDialog.Builder(this)
            .setMessage("Xóa toàn bộ lượt điểm danh của ${repository.currentSession()}?")
            .setNegativeButton("Hủy", null)
            .setPositiveButton("Xóa") { _, _ -> repository.clearSession(); refresh() }.show()
    }

    private fun confirmRemove(id: String, name: String) {
        AlertDialog.Builder(this).setMessage("Xóa $name ($id) và embedding đã lưu?")
            .setNegativeButton("Hủy", null)
            .setPositiveButton("Xóa") { _, _ -> repository.removeStudent(id); refresh() }.show()
    }

    private fun exportAttendance(uri: Uri) {
        runCatching {
            contentResolver.openOutputStream(uri)?.use { stream ->
                OutputStreamWriter(stream, Charsets.UTF_8).use { writer ->
                    writer.write("session,student_id,student_name,check_in_time,status,similarity\n")
                    repository.attendance().filter { it.session == repository.currentSession() }
                        .forEach { record ->
                            val csvName = "\"${record.studentName.replace("\"", "\"\"")}\""
                            writer.write(
                                "${record.session},${record.studentId},$csvName,${
                                    SimpleDateFormat(
                                        "yyyy-MM-dd HH:mm:ss",
                                        Locale.US
                                    ).format(Date(record.timestamp))
                                },${record.status},${record.similarity}\n"
                            )
                        }
                }
            } ?: error("Không mở được file")
        }.onSuccess { toast("Đã xuất CSV điểm danh.") }
            .onFailure { toast("Xuất CSV lỗi: ${it.localizedMessage}") }
    }

    private fun toast(message: String) = Toast.makeText(this, message, Toast.LENGTH_LONG).show()

    private fun enrollmentPose(face: com.google.mlkit.vision.face.Face): String {
        val yaw = face.headEulerAngleY
        return when {
            kotlin.math.abs(yaw) < 12f -> "FRONT"
            yaw < 0f -> "LEFT"
            else -> "RIGHT"
        }
    }

    override fun onDestroy() {
        studentsDetector.close()
        if (::executor.isInitialized) executor.shutdown()
        embedder?.close()
        super.onDestroy()
    }
}
