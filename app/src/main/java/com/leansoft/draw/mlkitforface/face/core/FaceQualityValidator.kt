package com.leansoft.draw.mlkitforface.face.core

import android.graphics.Bitmap
import com.google.mlkit.vision.face.Face
import kotlin.math.abs
import androidx.core.graphics.get

/** Lightweight demo quality gates; they reject obviously unsuitable samples before embedding. */
object FaceQualityValidator {
    fun rejectionReason(bitmap: Bitmap, face: Face, allowEnrollmentPose: Boolean = false): String? {
        val bounds = face.boundingBox
        if (bounds.width() < bitmap.width * MIN_FACE_RATIO || bounds.height() < bitmap.height * MIN_FACE_RATIO) {
            return "khuôn mặt quá nhỏ"
        }
        val maxYaw = if (allowEnrollmentPose) ENROLLMENT_MAX_YAW_DEGREES else MAX_YAW_DEGREES
        val maxPitch = if (allowEnrollmentPose) ENROLLMENT_MAX_PITCH_DEGREES else MAX_PITCH_DEGREES
        if (abs(face.headEulerAngleY) > maxYaw || abs(face.headEulerAngleX) > maxPitch) {
            return "góc nhìn quá nghiêng (yaw ${face.headEulerAngleY.toInt()}°, pitch ${face.headEulerAngleX.toInt()}°)"
        }
        val crop = runCatching { FaceImagePreprocessor.cropAndAlign(bitmap, face) }.getOrNull()
            ?: return "không cắt được vùng khuôn mặt"
        return try {
            if (laplacianVariance(crop) < MIN_LAPLACIAN_VARIANCE) "ảnh khuôn mặt bị mờ" else null
        } finally {
            if (crop !== bitmap) crop.recycle()
        }
    }

    private fun laplacianVariance(bitmap: Bitmap): Double {
        // Sample a small grayscale grid and measure local Laplacian contrast as a blur proxy.
        val width = bitmap.width
        val height = bitmap.height
        if (width < 5 || height < 5) return 0.0
        val stepX = (width / GRID_SIZE).coerceAtLeast(1)
        val stepY = (height / GRID_SIZE).coerceAtLeast(1)
        val values = ArrayList<Double>()
        fun gray(x: Int, y: Int): Double {
            val color = bitmap[x, y]
            return 0.299 * ((color shr 16) and 0xff) + 0.587 * ((color shr 8) and 0xff) + 0.114 * (color and 0xff)
        }
        var y = 1
        while (y < height - 1) {
            var x = 1
            while (x < width - 1) {
                values += 4 * gray(x, y) - gray(x - stepX.coerceAtMost(x), y) - gray((x + stepX).coerceAtMost(width - 1), y) -
                    gray(x, y - stepY.coerceAtMost(y)) - gray(x, (y + stepY).coerceAtMost(height - 1))
                x += stepX
            }
            y += stepY
        }
        if (values.isEmpty()) return 0.0
        val mean = values.average()
        return values.sumOf { (it - mean) * (it - mean) } / values.size
    }

    private const val MIN_FACE_RATIO = 0.16f
    private const val MAX_YAW_DEGREES = 28f
    private const val MAX_PITCH_DEGREES = 25f
    const val ENROLLMENT_MAX_YAW_DEGREES = 35f
    const val ENROLLMENT_MAX_PITCH_DEGREES = 30f
    private const val MIN_LAPLACIAN_VARIANCE = 35.0
    private const val GRID_SIZE = 48
}
