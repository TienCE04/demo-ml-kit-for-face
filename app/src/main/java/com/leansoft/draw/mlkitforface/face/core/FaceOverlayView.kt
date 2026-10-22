package com.leansoft.draw.mlkitforface.face.core

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View

class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF7EF0C6.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 4f * resources.displayMetrics.density
    }
    private var boxes = emptyList<Rect>()
    private var imageWidth = 1
    private var imageHeight = 1

    fun setFaces(newBoxes: List<Rect>, width: Int, height: Int) {
        boxes = newBoxes.map(::Rect)
        imageWidth = width.coerceAtLeast(1)
        imageHeight = height.coerceAtLeast(1)
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (boxes.isEmpty() || width == 0 || height == 0) return
        // PreviewView uses center crop; mirror the front camera's visible preview.
        val scale = maxOf(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        val offsetX = (width - imageWidth * scale) / 2f
        val offsetY = (height - imageHeight * scale) / 2f
        boxes.forEach { box ->
            val left = width - (box.right * scale + offsetX)
            val right = width - (box.left * scale + offsetX)
            canvas.drawRoundRect(left, box.top * scale + offsetY, right, box.bottom * scale + offsetY, 18f, 18f, paint)
        }
    }
}
