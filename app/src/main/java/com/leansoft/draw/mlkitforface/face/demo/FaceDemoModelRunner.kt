package com.leansoft.draw.mlkitforface.face.demo

import android.content.Context
import android.graphics.Bitmap
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.leansoft.draw.mlkitforface.face.core.FaceEmbeddingModel
import com.leansoft.draw.mlkitforface.face.core.FaceNetEmbeddingModel

/** Owns detector/model instances so the Activity does not manage inference resources. */
class FaceDemoModelRunner(context: Context) : AutoCloseable {
    private val context = context.applicationContext
    private var detector: FaceDetector? = null
    private var mobileFaceNet: FaceEmbeddingModel? = null
    private var faceNet: FaceNetEmbeddingModel? = null

    fun detector(): FaceDetector {
        detector?.let { return it }
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setMinFaceSize(0.12f)
            .build()
        return FaceDetection.getClient(options).also { detector = it }
    }

    fun embed(face: Bitmap, model: ModelChoice): FloatArray = when (model) {
        ModelChoice.MOBILE_FACE_NET -> mobileFaceNet().embed(face)
        ModelChoice.FACE_NET -> faceNet().embed(face)
    }

    override fun close() {
        detector?.close()
        mobileFaceNet?.close()
        faceNet?.close()
        detector = null
        mobileFaceNet = null
        faceNet = null
    }

    private fun mobileFaceNet(): FaceEmbeddingModel =
        mobileFaceNet ?: FaceEmbeddingModel(context).also { mobileFaceNet = it }

    private fun faceNet(): FaceNetEmbeddingModel =
        faceNet ?: FaceNetEmbeddingModel(context).also { faceNet = it }
}
