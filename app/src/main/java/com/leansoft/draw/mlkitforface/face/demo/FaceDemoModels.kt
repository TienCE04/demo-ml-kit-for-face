package com.leansoft.draw.mlkitforface.face.demo

import android.graphics.Bitmap
import android.net.Uri

enum class CaptureTarget {
    ENROLLMENT,
    LIVE_LIVENESS,
    SINGLE_PHOTO
}

enum class ModelChoice(val label: String, val threshold: Float) {
    MOBILE_FACE_NET("MobileFaceNet", 0.72f),
    FACE_NET("FaceNet", 0.72f)
}

data class EnrollmentSource(
    val bitmap: Bitmap? = null,
    val uri: Uri? = null
)

data class EnrollmentSample(
    val index: Int,
    val vector: FloatArray?,
    val error: String? = null
)

data class EmbeddingResult(
    val vectors: List<FloatArray>,
    val centroid: FloatArray,
    val rejected: Int
)

object FaceDemoRules {
    const val MIN_ENROLLMENT_PHOTOS = 3
    const val MAX_ENROLLMENT_PHOTOS = 4
    const val FRONT_YAW_DEGREES = 12f
    val LIVE_POSES = listOf("FRONT", "LEFT", "RIGHT")
}
