package com.leansoft.draw.mlkitforface

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.face.Face
import kotlin.math.roundToInt

object FaceImagePreprocessor {

    fun loadCorrectly(
        context: Context,
        uri: Uri
    ): Bitmap {
        val resolver = context.contentResolver

        val bitmap = resolver
            .openInputStream(uri)
            ?.use { BitmapFactory.decodeStream(it) }
            ?: error("Không đọc được ảnh")

        val orientation = resolver
            .openInputStream(uri)
            ?.use { inputStream ->
                ExifInterface(inputStream).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            }
            ?: ExifInterface.ORIENTATION_NORMAL

        val matrix = Matrix()

        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> {
                matrix.postRotate(90f)
            }

            ExifInterface.ORIENTATION_ROTATE_180 -> {
                matrix.postRotate(180f)
            }

            ExifInterface.ORIENTATION_ROTATE_270 -> {
                matrix.postRotate(270f)
            }

            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> {
                matrix.preScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                matrix.preScale(1f, -1f)
            }
        }

        val corrected = try {
            Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.width,
                bitmap.height,
                matrix,
                true
            )
        } catch (error: Exception) {
            bitmap.recycle()
            throw error
        }
        if (corrected !== bitmap) bitmap.recycle()
        return corrected
    }

    fun cropAndAlign(
        bitmap: Bitmap,
        face: Face,
        marginRatio: Float = 0.25f
    ): Bitmap {
        val bounds = Rect(face.boundingBox)

        val marginX =
            (bounds.width() * marginRatio).roundToInt()

        val marginY =
            (bounds.height() * marginRatio).roundToInt()

        val left = (bounds.left - marginX)
            .coerceAtLeast(0)

        val top = (bounds.top - marginY)
            .coerceAtLeast(0)

        val right = (bounds.right + marginX)
            .coerceAtMost(bitmap.width)

        val bottom = (bounds.bottom + marginY)
            .coerceAtMost(bitmap.height)

        val crop = Bitmap.createBitmap(
            bitmap,
            left,
            top,
            right - left,
            bottom - top
        )

        // Xoay khuôn mặt theo roll để mặt gần thẳng hơn
        val matrix = Matrix().apply {
            postRotate(-face.headEulerAngleZ)
        }

        val aligned = Bitmap.createBitmap(
            crop,
            0,
            0,
            crop.width,
            crop.height,
            matrix,
            true
        )
        if (crop !== bitmap && aligned !== crop) crop.recycle()
        return aligned
    }
}
