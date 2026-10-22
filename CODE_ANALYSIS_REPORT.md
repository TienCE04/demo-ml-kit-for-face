# Báo cáo phân tích logic ứng dụng nhận diện khuôn mặt

Ngày cập nhật: 22/10/2026  
Phạm vi hiện tại: màn hình demo embedding chạy tạm thời trong RAM. Luồng chính là **chụp hoặc chọn 3–4 ảnh → chọn model → tạo embedding → camera liveness realtime FRONT/LEFT/RIGHT → xác thực**.

## 1. Luồng hiện tại

```text
Chụp hoặc chọn 3–4 ảnh đăng ký
        ↓
ML Kit phát hiện đúng 1 mặt + quality gate
        ↓
Crop/align khuôn mặt
        ↓
Model được chọn tạo embedding trong RAM
        ↓
Chọn chế độ xác thực
        ├─ Camera realtime → ML Kit kiểm tra FRONT → LEFT → RIGHT
        │                 → embedding frame camera cuối
        └─ Chụp một ảnh  → ML Kit + embedding ảnh vừa chụp
                           ↓
             Cosine với các vector đăng ký → MATCH / NO MATCH
```

Sau khi tạo embedding, người dùng chọn một trong hai chế độ xác thực:

1. Camera realtime: chạy liveness FRONT → LEFT → RIGHT rồi so khớp frame camera cuối.
2. Chụp một ảnh: bỏ qua liveness và so khớp ngay ảnh vừa chụp với embedding.

Màn hình này không nhập mã sinh viên, không nhập tên, không yêu cầu consent và không gọi `AttendanceRepository`. Ảnh chọn từ thiết bị, bitmap camera và vector chỉ tồn tại trong vòng đời `FaceDemoActivity`; khi Activity bị hủy, chúng được giải phóng.

Các màn hình điểm danh cũ và repository vẫn còn trong project để giữ tương thích với phần app cũ, nhưng không được gọi bởi luồng demo mới.

`FaceDemoActivity` đã được đặt làm launcher trong `AndroidManifest.xml`, còn `MainActivity` cũ không còn là launcher. Vì vậy khi mở app, người dùng đi thẳng vào luồng tối giản này.

## 2. Các file cần đọc

| File | Vai trò |
|---|---|
| [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt) | Điều phối toàn bộ luồng chụp, chọn model, embedding và liveness |
| [activity_face_demo.xml](app/src/main/res/layout/activity_face_demo.xml) | Giao diện tối giản gồm 3 bước |
| [FaceEmbeddingModel.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/core/FaceEmbeddingModel.kt) | Wrapper MobileFaceNet |
| [FaceNetEmbeddingModel.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/core/FaceNetEmbeddingModel.kt) | Wrapper FaceNet |
| [FaceQualityValidator.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/core/FaceQualityValidator.kt) | Loại ảnh quá nhỏ, nghiêng hoặc mờ |
| [FaceImagePreprocessor.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/core/FaceImagePreprocessor.kt) | Crop và xoay khuôn mặt trước inference |
| [FaceDemoModelRunner.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoModelRunner.kt) | Quản lý detector và model lazy |
| [FaceEmbeddingProcessor.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceEmbeddingProcessor.kt) | Pipeline enrollment và verification |
| [RealtimeLivenessController.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/RealtimeLivenessController.kt) | CameraX analyzer cho liveness |

Manifest launcher: [AndroidManifest.xml](app/src/main/AndroidManifest.xml).

Cấu trúc mới tách theo trách nhiệm:

- `face/demo/FaceDemoActivity.kt`: chỉ điều phối UI, trạng thái phiên và callback.
- `face/demo/FaceDemoModels.kt`: enum model, nguồn ảnh và luật tối thiểu 3–4 ảnh.
- `face/demo/FaceDemoModelRunner.kt`: khởi tạo detector và wrapper model theo kiểu lazy.
- `face/demo/FaceEmbeddingProcessor.kt`: phát hiện mặt, quality gate, crop/align, embedding và cosine.
- `face/demo/RealtimeLivenessController.kt`: CameraX + ML Kit cho liveness realtime.
- `face/core/`: MobileFaceNet, FaceNet, preprocessing, quality validator và overlay.
- `attendance/`: repository, gateway và màn hình điểm danh cũ.
- `legacy/MainActivity.kt`: màn hình cũ có BLE, không nằm trong launcher hiện tại.

Activity không còn giữ detector, interpreter, CameraX analyzer và toàn bộ logic inference trong cùng một file.

## 3. Giao diện hiện tại

Layout chỉ còn ba card:

1. `CHỤP HOẶC CHỌN 3–4 ẢNH`: nút `btnCaptureEnrollmentPhoto` và `btnPickEnrollmentPhotos`.
2. `CHỌN MODEL VÀ TẠO EMBEDDING`: nút `btnSelectModel` và `btnCreateEmbedding`.
3. `XÁC THỰC`: preview `livenessPreview`, nút `btnStartLiveness` cho camera realtime, `btnCaptureProbePhoto` cho một ảnh và `btnStopLiveness`.

Xem [activity_face_demo.xml](app/src/main/res/layout/activity_face_demo.xml), [activity_face_demo.xml](app/src/main/res/layout/activity_face_demo.xml) và [activity_face_demo.xml](app/src/main/res/layout/activity_face_demo.xml).

## 4. Bộ nhớ tạm, không lưu hồ sơ

Các biến giữ dữ liệu của phiên hiện tại:

```kotlin
private val enrollmentSources = mutableListOf<EnrollmentSource>()
private var enrollmentVectors: List<FloatArray> = emptyList()
private var enrollmentCentroid: FloatArray? = null
```

Xem [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt).

- `enrollmentSources`: bitmap vừa chụp hoặc `Uri` ảnh chọn từ thiết bị.
- `enrollmentVectors`: embedding của các ảnh hợp lệ.
- `enrollmentCentroid`: vector trung bình dùng để xác nhận trạng thái đã tạo embedding.

Không có `AttendanceRepository`, `saveStudent()`, `SharedPreferences` hay file ảnh trong luồng này. Khi chụp thêm ảnh, embedding cũ bị xóa bằng `clearEmbedding()` để không trộn model hoặc trộn bộ ảnh cũ với bộ ảnh mới.

```kotlin
enrollmentSources += EnrollmentSource(bitmap = bitmap)
clearEmbedding()
showEnrollmentStatus("Đã có ${enrollmentSources.size}/4 ảnh...")
```

Xem [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt).

## 5. Chụp hoặc chọn tối thiểu 3 ảnh

```kotlin
if (enrollmentSources.size >= MAX_ENROLLMENT_PHOTOS) {
    bitmap.recycle()
    showEnrollmentStatus("Đã đủ 4 ảnh.")
    return
}
```

`MAX_ENROLLMENT_PHOTOS = 4`, còn `MIN_ENROLLMENT_PHOTOS = 3`. Nút chụp bị khóa khi đã đủ bốn ảnh. Xem [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt) và [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt).

Nút chụp dùng `TakePicturePreview()`. Nút chọn dùng `GetMultipleContents()` và chỉ giữ `Uri` trong phiên hiện tại. Code không nén ảnh vào `cacheDir` và không ghi hồ sơ ra đĩa. Nếu chọn ảnh, số lượng phải từ 3 đến 4:

```kotlin
if (uris.size !in MIN_ENROLLMENT_PHOTOS..MAX_ENROLLMENT_PHOTOS) {
    showEnrollmentStatus("Hãy chọn từ 3 đến 4 ảnh.")
    return@registerForActivityResult
}
enrollmentSources += uris.map { EnrollmentSource(uri = it) }
```

Xem [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt).

## 6. Xin quyền camera đúng theo mục đích

```kotlin
private var pendingCapture = CaptureTarget.ENROLLMENT
```

```kotlin
when (pendingCapture) {
    CaptureTarget.ENROLLMENT -> captureEnrollmentPhoto.launch(null)
    CaptureTarget.LIVE_LIVENESS -> startLivenessCamera()
    CaptureTarget.SINGLE_PHOTO -> captureProbePhoto.launch(null)
}
```

Xem [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt) và [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt).

`pendingCapture` bảo đảm nếu người dùng cấp quyền camera lần đầu từ nút đăng ký thì app mở đúng camera đăng ký; nếu cấp quyền từ nút liveness thì app mở đúng CameraX preview realtime.

Với lựa chọn chụp một ảnh, `pendingCapture = SINGLE_PHOTO` mở `TakePicturePreview()` và không khởi động ImageAnalysis.

## 7. Nút chọn model

```kotlin
private var selectedModel = ModelChoice.MOBILE_FACE_NET
```

```kotlin
enum class ModelChoice(val label: String, val threshold: Float) {
    MOBILE_FACE_NET("MobileFaceNet", 0.72f),
    FACE_NET("FaceNet", 0.72f)
}
```

Xem [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt) và [FaceDemoModels.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoModels.kt).

Nút `btnSelectModel` mở hộp chọn:

```kotlin
val choices = ModelChoice.values().map { it.label }.toTypedArray()
AlertDialog.Builder(this)
    .setTitle("Chọn model embedding")
    .setSingleChoiceItems(choices, selectedModel.ordinal) { dialog, which ->
        selectedModel = ModelChoice.values()[which]
        clearEmbedding()
        dialog.dismiss()
    }
    .show()
```

Xem [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt).

Khi đổi model, embedding cũ bị xóa bắt buộc tạo lại. Điều này tránh việc dùng vector MobileFaceNet để so với vector FaceNet.

## 8. Model thực sự được tải ở đâu

MobileFaceNet tải file `mobile_face_net.tflite` trong [FaceEmbeddingModel.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/core/FaceEmbeddingModel.kt). FaceNet tải file `facenet_512_int_quantized.tflite` trong [FaceNetEmbeddingModel.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/core/FaceNetEmbeddingModel.kt).

`FaceDemoModelRunner` giữ detector và hai wrapper theo kiểu lazy. Model chỉ được khởi tạo khi người dùng chọn model đó và bấm tạo embedding hoặc xác thực:

```kotlin
fun embed(face: Bitmap, model: ModelChoice): FloatArray = when (model) {
    ModelChoice.MOBILE_FACE_NET -> mobileFaceNet().embed(face)
    ModelChoice.FACE_NET -> faceNet().embed(face)
}
```

Xem [FaceDemoModelRunner.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoModelRunner.kt). Khi đổi model, Activity xóa embedding cũ và bắt buộc tạo lại để không so sánh vector ở hai không gian khác nhau.

## 9. Tạo embedding sau khi đủ ảnh

Activity chỉ kiểm tra số lượng ảnh rồi chuyển phần xử lý nặng sang `worker`:

```kotlin
worker.execute {
    val result = embeddingProcessor.createEmbedding(sources, model)
}
```

Xem [FaceDemoActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceDemoActivity.kt) và [FaceEmbeddingProcessor.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceEmbeddingProcessor.kt). Processor xử lý mỗi ảnh theo thứ tự:

1. Đọc bitmap từ camera hoặc `Uri`.
2. ML Kit tìm khuôn mặt và bắt buộc đúng một khuôn mặt.
3. `FaceQualityValidator` kiểm tra kích thước, góc và độ nét.
4. `FaceImagePreprocessor.cropAndAlign()` căn chỉnh khuôn mặt.
5. Model đã chọn tạo vector.
6. Bitmap trung gian được recycle.

Processor trả về các vector hợp lệ, centroid và số ảnh bị loại. Nếu dưới 3 ảnh hợp lệ, pipeline thất bại; nếu có 4 ảnh nhưng chỉ 3 ảnh hợp lệ, app vẫn tiếp tục với 3 ảnh đó.

## 10. MobileFaceNet preprocessing

MobileFaceNet nhận input `[1, 112, 112, 3]` và output `[1, 192]`:

```kotlin
require(interpreter.getInputTensor(0).shape()
    .contentEquals(intArrayOf(1, INPUT_SIZE, INPUT_SIZE, 3)))
require(interpreter.getOutputTensor(0).shape()
    .contentEquals(intArrayOf(1, EMBEDDING_SIZE)))
```

Xem [FaceEmbeddingModel.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/core/FaceEmbeddingModel.kt).

Mỗi pixel RGB được đổi về gần `[-1, 1]`:

```kotlin
input.putFloat(((red / 128f) - 1f))
input.putFloat(((green / 128f) - 1f))
input.putFloat(((blue / 128f) - 1f))
```

Sau inference, vector được L2-normalize trước khi tính cosine. Xem [FaceEmbeddingModel.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/core/FaceEmbeddingModel.kt).

## 11. FaceNet preprocessing

FaceNet wrapper đọc kích thước tensor trực tiếp từ model:

```kotlin
val shape = inputTensor.shape()
require(shape.size == 4 && shape[0] == 1 && shape[3] == 3)
inputSize = shape[1]
```

Xem [FaceNetEmbeddingModel.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/core/FaceNetEmbeddingModel.kt).

Wrapper hỗ trợ `FLOAT32`, `INT8` và `UINT8`, sau đó normalize output. Model phụ chỉ được chạy khi người dùng chọn FaceNet; không chạy song song với MobileFaceNet trong phiên tối giản này.

## 12. Liveness bằng camera realtime

Liveness yêu cầu ba tư thế `FRONT → LEFT → RIGHT`, được định nghĩa trong `FaceDemoRules.LIVE_POSES`. Activity quản lý bước hiện tại và thông báo cho người dùng; phần CameraX/ML Kit nằm trong [RealtimeLivenessController.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/RealtimeLivenessController.kt).

Controller mở camera trước, hiển thị `PreviewView`, dùng `ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST` và chỉ gửi pose về Activity khi ML Kit phát hiện đúng một khuôn mặt. Activity chuyển yaw thành pose: yaw gần 0 là `FRONT`, yaw âm là `LEFT`, yaw dương là `RIGHT`.

Khi đủ ba pose, Activity lấy một bản sao frame hiện tại từ `PreviewView`, dừng camera rồi gọi `FaceEmbeddingProcessor.verifyLiveBitmap()`. Processor phát hiện mặt lần nữa, crop/align, tạo embedding bằng model đã chọn, tính cosine với các vector đăng ký và trả về kết quả kèm chuỗi liveness.

Đây là active liveness heuristic dựa trên việc đổi góc mặt trước camera, chưa phải anti-spoof model chuyên dụng.

## 13. Hai chế độ xác thực

### 13.1. Camera realtime

Camera realtime chạy liveness trước rồi mới so khớp frame cuối. Kết quả nằm trong `verifyLiveBitmap()` của [FaceEmbeddingProcessor.kt](app/src/main/java/com/leansoft/draw/mlkitforface/face/demo/FaceEmbeddingProcessor.kt).

### 13.2. Chụp một ảnh để so khớp

Nút `btnCaptureProbePhoto` mở `TakePicturePreview()` một lần. Bitmap được chuyển vào `verifySinglePhoto()`, không khởi động CameraX và không chạy liveness. Processor vẫn yêu cầu đúng một khuôn mặt, quality gate, crop/align và model đã chọn trước khi tính cosine.

```kotlin
val score = enrollmentVectors.maxOf {
    FaceEmbeddingModel.cosine(probe, it)
}
val passed = score >= model.threshold
```

Kết quả hiển thị `MATCH/NO MATCH`, model và cosine. Bitmap probe được recycle sau khi xử lý.

## 14. Xác thực bằng cosine

Frame camera cuối hoặc ảnh probe được so với tất cả vector đăng ký; điểm cao nhất được dùng làm score. Threshold hiện tại là giá trị demo `0.72` cho cả hai model. Muốn đánh giá nghiêm túc cần hiệu chỉnh threshold riêng bằng tập ảnh genuine/impostor. Luồng này không tạo attendance record.

## 15. Giải phóng dữ liệu

Khi Activity đóng:

- Bitmap đăng ký được recycle.
- `RealtimeLivenessController` unbind camera.
- `FaceDemoModelRunner` đóng ML Kit detector, MobileFaceNet interpreter và FaceNet interpreter.
- Executor được shutdown.

Không có dữ liệu embedding, ảnh hoặc thông tin cá nhân nào được ghi vào repository trong luồng demo.

## 16. Các giới hạn cần nêu

1. `TakePicturePreview()` chỉ trả ảnh preview, không phải ảnh full-resolution; ảnh chọn từ thiết bị có thể có kích thước khác nhau.
2. Chế độ chụp một ảnh không có liveness; chỉ camera realtime mới kiểm tra đổi góc mặt.
3. Liveness hiện là thử thách đổi góc mặt trong camera realtime, chưa có anti-spoof neural network.
4. Dấu yaw trái/phải phụ thuộc cách ML Kit biểu diễn camera trước; nếu thiết bị đảo chiều, cần đổi nhãn LEFT/RIGHT.
5. Frame cuối lấy từ `PreviewView` có thể có độ phân giải thấp hơn ảnh gốc.
6. Threshold `0.72` chỉ là giá trị demo, chưa được đo trên dữ liệu thực tế.
7. Khi đổi model, phải tạo embedding lại vì không được so vector khác không gian.
8. Dữ liệu chỉ tồn tại trong Activity; thoát màn hình hoặc app là mất toàn bộ phiên.
9. Màn hình điểm danh/repository cũ vẫn tồn tại ngoài luồng demo tối giản này.

## 17. Kiểm tra build

Đã chạy:

```text
GRADLE_USER_HOME=/private/tmp/demo-gradle-home bash ./gradlew --no-daemon :app:compileDebugKotlin
```

Kết quả:

```text
BUILD SUCCESSFUL
```

Chỉ còn các cảnh báo deprecation từ màn hình legacy [MainActivity.kt](app/src/main/java/com/leansoft/draw/mlkitforface/legacy/MainActivity.kt) về CameraX và BLE; luồng mới biên dịch thành công.
