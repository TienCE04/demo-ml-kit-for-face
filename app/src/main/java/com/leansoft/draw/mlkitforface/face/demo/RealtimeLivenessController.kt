package com.leansoft.draw.mlkitforface.face.demo

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetector
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean

/** CameraX + ML Kit frame loop. The Activity only receives detected poses/errors. */
class RealtimeLivenessController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val detector: FaceDetector,
    private val analyzerExecutor: Executor,
    private val poseForYaw: (Float) -> String,
    private val onFrame: (String) -> Unit,
    private val onError: (String) -> Unit
) : AutoCloseable {
    private var provider: ProcessCameraProvider? = null
    private val processing = AtomicBoolean(false)

    fun start() {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            runCatching {
                val cameraProvider = future.get()
                provider = cameraProvider
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(analyzerExecutor, ::analyze)
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    analysis
                )
            }.onFailure { onError(it.localizedMessage ?: "không mở được camera") }
        }, ContextCompat.getMainExecutor(context))
    }

    override fun close() {
        provider?.unbindAll()
        provider = null
        processing.set(false)
    }

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    private fun analyze(proxy: ImageProxy) {
        val mediaImage = proxy.image
        if (mediaImage == null || !processing.compareAndSet(false, true)) {
            proxy.close()
            return
        }
        detector.process(InputImage.fromMediaImage(mediaImage, proxy.imageInfo.rotationDegrees))
            .addOnSuccessListener { faces ->
                if (faces.size == 1) onFrame(poseForYaw(faces.single().headEulerAngleY))
            }
            .addOnFailureListener { onError(it.localizedMessage ?: "ML Kit không xử lý được frame") }
            .addOnCompleteListener {
                processing.set(false)
                proxy.close()
            }
    }
}
