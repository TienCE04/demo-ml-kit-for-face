package com.leansoft.draw.mlkitforface.face.demo

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.leansoft.draw.mlkitforface.face.core.FaceEmbeddingModel
import com.leansoft.draw.mlkitforface.face.core.FaceImagePreprocessor
import com.leansoft.draw.mlkitforface.face.core.FaceQualityValidator
import java.util.Locale

/** Pure enrollment/probe pipeline. It never writes a student or embedding to storage. */
class FaceEmbeddingProcessor(
    private val context: Context,
    private val models: FaceDemoModelRunner
) {
    fun createEmbedding(sources: List<EnrollmentSource>, model: ModelChoice): EmbeddingResult {
        val samples = sources.mapIndexed { index, source ->
            createEnrollmentSample(index + 1, source, model)
        }
        val valid = samples.mapNotNull { it.vector }
        require(valid.size >= FaceDemoRules.MIN_ENROLLMENT_PHOTOS) {
            "Chỉ có ${valid.size}/${sources.size} ảnh đạt. Hãy chọn/chụp lại ảnh rõ mặt hơn."
        }
        return EmbeddingResult(
            vectors = valid,
            centroid = FaceEmbeddingModel.centroid(valid),
            rejected = samples.count { it.vector == null }
        )
    }

    fun verifyLiveBitmap(
        bitmap: Bitmap,
        model: ModelChoice,
        enrollmentVectors: List<FloatArray>,
        livenessPoses: List<String>
    ): String = verifyBitmap(bitmap, model, enrollmentVectors) { score ->
        buildString {
            appendLine(if (score >= model.threshold) "XÁC THỰC THÀNH CÔNG" else "XÁC THỰC THẤT BẠI")
            appendLine("Model: ${model.label}")
            appendLine("Liveness: PASS · ${livenessPoses.joinToString(" → ")}")
            appendLine("Cosine: ${score.fmt()} / ngưỡng ${model.threshold.fmt()}")
            appendLine("Ảnh, thông tin cá nhân và embedding không được lưu.")
        }
    }

    fun verifySinglePhoto(
        bitmap: Bitmap,
        model: ModelChoice,
        enrollmentVectors: List<FloatArray>
    ): String = verifyBitmap(bitmap, model, enrollmentVectors) { score ->
        buildString {
            appendLine(if (score >= model.threshold) "SO KHỚP: MATCH" else "SO KHỚP: NO MATCH")
            appendLine("Model: ${model.label}")
            appendLine("Cosine: ${score.fmt()} / ngưỡng ${model.threshold.fmt()}")
            appendLine("Chế độ này chỉ so một ảnh; không chạy liveness.")
            appendLine("Ảnh và embedding không được lưu.")
        }
    }

    private fun createEnrollmentSample(
        index: Int,
        source: EnrollmentSource,
        model: ModelChoice
    ): EnrollmentSample {
        var loaded: Bitmap? = null
        return try {
            val bitmap = source.bitmap
                ?: FaceImagePreprocessor.loadCorrectly(context, source.uri!!).also { loaded = it }
            val faces = Tasks.await(models.detector().process(InputImage.fromBitmap(bitmap, 0)))
            require(faces.size == 1) { "Ảnh $index cần đúng 1 khuôn mặt; phát hiện ${faces.size}." }
            val face = faces.single()
            val rejection = FaceQualityValidator.rejectionReason(bitmap, face, allowEnrollmentPose = true)
            require(rejection == null) { "Ảnh $index không đạt: $rejection" }
            val aligned = FaceImagePreprocessor.cropAndAlign(bitmap, face)
            try {
                EnrollmentSample(index, models.embed(aligned, model))
            } finally {
                aligned.recycle()
            }
        } catch (error: Exception) {
            EnrollmentSample(index, null, error.localizedMessage ?: "không xử lý được ảnh")
        } finally {
            loaded?.let { if (!it.isRecycled) it.recycle() }
        }
    }

    private fun verifyBitmap(
        bitmap: Bitmap,
        model: ModelChoice,
        enrollmentVectors: List<FloatArray>,
        report: (Float) -> String
    ): String {
        return try {
            val faces = Tasks.await(models.detector().process(InputImage.fromBitmap(bitmap, 0)))
            require(faces.size == 1) { "Ảnh cần đúng 1 khuôn mặt; phát hiện ${faces.size}." }
            val face = faces.single()
            val rejection = FaceQualityValidator.rejectionReason(bitmap, face, allowEnrollmentPose = true)
            require(rejection == null) { "Ảnh không đạt: $rejection" }
            val aligned = FaceImagePreprocessor.cropAndAlign(bitmap, face)
            val probe = try {
                models.embed(aligned, model)
            } finally {
                aligned.recycle()
            }
            val score = enrollmentVectors.maxOf { FaceEmbeddingModel.cosine(probe, it) }
            report(score)
        } finally {
            bitmap.recycle()
        }
    }

    private fun Float.fmt(): String = String.format(Locale.US, "%.4f", this)
}
