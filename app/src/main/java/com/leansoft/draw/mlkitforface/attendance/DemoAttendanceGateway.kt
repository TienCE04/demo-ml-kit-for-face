package com.leansoft.draw.mlkitforface.attendance
import com.leansoft.draw.mlkitforface.face.core.FaceEmbeddingModel

import org.json.JSONObject
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64
import java.util.UUID

/**
 * Local server-shaped gateway used by the demo.
 *
 * The production contract in project3 makes the API server the authority for
 * BLE evidence, face similarity and the final attendance result. This class
 * keeps that boundary visible while the sample app runs without a backend:
 * token verification, attempt idempotency, challenge expiry and matching all
 * happen here instead of in the Activity UI.
 */
class DemoAttendanceGateway(private val repository: AttendanceRepository) {
    data class Broadcast(
        val sessionCode: String,
        val token: String,
        val tokenFingerprint: String,
        val expiresAt: Long,
        val rotateEveryMs: Long = DEFAULT_ROTATE_MS
    )

    data class Attempt(
        val id: String,
        val clientAttemptId: String,
        val sessionCode: String,
        var status: String = "RECEIVED",
        var expiresAt: Long = 0L,
        var bleEvidence: BleEvidence? = null,
        var challenge: FaceChallenge? = null
    )

    data class BleEvidence(
        val sessionCode: String,
        val token: String,
        val scanCount: Int,
        val scanDurationMs: Long,
        val rssiMin: Int,
        val rssiMax: Int,
        val rssiAverage: Int,
        val firstSeenAt: Long,
        val lastSeenAt: Long
    )

    data class FaceChallenge(
        val nonce: String,
        val type: String,
        val expiresAt: Long,
        var used: Boolean = false
    )

    data class BleResult(
        val accepted: Boolean,
        val code: String,
        val message: String,
        val attempt: Attempt? = null
    )

    data class FaceResult(
        val accepted: Boolean,
        val code: String,
        val message: String,
        val student: EnrolledStudent? = null,
        val similarity: Float = 0f,
        val margin: Float = 0f,
        val attendance: String? = null,
        val livenessPassed: Boolean = false,
        val attempt: Attempt? = null
    )

    private val random = SecureRandom()
    private val broadcasts = mutableMapOf<String, Broadcast>()
    private val attemptsByClientId = mutableMapOf<String, Attempt>()
    private val attemptsById = mutableMapOf<String, Attempt>()

    @Synchronized
    fun startSession(sessionCode: String): Broadcast = issueToken(sessionCode.trim())

    @Synchronized
    fun rotateToken(sessionCode: String): Broadcast? {
        if (!broadcasts.containsKey(sessionCode)) return null
        return issueToken(sessionCode)
    }

    @Synchronized
    fun stopSession(sessionCode: String) {
        broadcasts.remove(sessionCode)
        attemptsByClientId.values.filter { it.sessionCode == sessionCode && it.status != "VERIFIED" }
            .forEach { it.status = "EXPIRED" }
    }

    @Synchronized
    fun currentBroadcast(sessionCode: String): Broadcast? = broadcasts[sessionCode]

    @Synchronized
    fun createAttempt(sessionCode: String, clientAttemptId: String): Attempt {
        attemptsByClientId[clientAttemptId]?.let { return it }
        val now = System.currentTimeMillis()
        val attempt = Attempt(
            id = UUID.randomUUID().toString(),
            clientAttemptId = clientAttemptId,
            sessionCode = sessionCode,
            expiresAt = now + ATTEMPT_TTL_MS
        )
        attemptsByClientId[clientAttemptId] = attempt
        attemptsById[attempt.id] = attempt
        return attempt
    }

    @Synchronized
    fun verifyBleEvidence(attemptId: String, evidence: BleEvidence): BleResult {
        val attempt = attemptsById[attemptId] ?: return rejected("ATTEMPT_NOT_FOUND", "Không tìm thấy attempt.")
        if (attempt.status == "VERIFIED") return BleResult(true, "BLE_VERIFIED", "Attempt đã được xác minh BLE.", attempt)
        if (attempt.expiresAt <= System.currentTimeMillis()) {
            attempt.status = "EXPIRED"
            return rejected("ATTEMPT_EXPIRED", "Attempt đã hết hạn.", attempt)
        }
        val broadcast = broadcasts[evidence.sessionCode]
            ?: return rejected("SESSION_NOT_ACTIVE", "Phiên điểm danh chưa phát BLE.", attempt)
        if (evidence.sessionCode != attempt.sessionCode) {
            return rejected("BLE_EVIDENCE_INVALID", "Beacon không thuộc phiên đang chọn.", attempt)
        }
        if (broadcast.expiresAt <= System.currentTimeMillis()) {
            return rejected("BLE_TOKEN_EXPIRED", "Token BLE đã hết hạn.", attempt)
        }
        if (!constantTimeEquals(hashToken(evidence.token), broadcast.tokenFingerprint)) {
            return rejected("BLE_TOKEN_INVALID", "Token BLE không hợp lệ.", attempt)
        }
        if (evidence.scanCount < MIN_SCAN_COUNT || evidence.scanDurationMs < MIN_SCAN_DURATION_MS) {
            return rejected("BLE_SCAN_INSUFFICIENT", "Chưa đủ số lần quét hoặc thời lượng quét.", attempt)
        }
        if (evidence.rssiAverage < MIN_RSSI) {
            return rejected("BLE_SIGNAL_TOO_WEAK", "Tín hiệu BLE quá yếu để xác minh.", attempt)
        }
        val now = System.currentTimeMillis()
        attempt.bleEvidence = evidence
        attempt.challenge = FaceChallenge(
            nonce = randomToken(24),
            type = "HOLD_STRAIGHT",
            expiresAt = now + FACE_CHALLENGE_TTL_MS
        )
        attempt.status = "VERIFYING"
        return BleResult(true, "BLE_VERIFIED", "BLE evidence hợp lệ; đã cấp face challenge.", attempt)
    }

    @Synchronized
    fun verifyFaceEvidence(
        attemptId: String,
        embedding: FloatArray,
        challengeNonce: String,
        livenessPassed: Boolean,
        sessionCode: String
    ): FaceResult {
        val attempt = attemptsById[attemptId] ?: return faceRejected("ATTEMPT_NOT_FOUND", "Không tìm thấy attempt.")
        val challenge = attempt.challenge
            ?: return faceRejected("BLE_REQUIRED", "Cần xác minh BLE trước khi mở camera.", attempt = attempt)
        if (attempt.status != "VERIFYING" || attempt.sessionCode != sessionCode) {
            return faceRejected("ATTEMPT_INVALID", "Attempt không ở bước xác minh khuôn mặt.", attempt = attempt)
        }
        if (challenge.used || challenge.expiresAt <= System.currentTimeMillis()) {
            attempt.status = "EXPIRED"
            return faceRejected("CHALLENGE_EXPIRED", "Face challenge đã hết hạn.", attempt = attempt)
        }
        if (!constantTimeEquals(challenge.nonce, challengeNonce)) {
            return faceRejected("CHALLENGE_INVALID", "Face challenge không hợp lệ.", attempt = attempt)
        }
        challenge.used = true
        if (!livenessPassed) {
            attempt.status = "REJECTED"
            return faceRejected("LIVENESS_FAILED", "Chưa hoàn thành kiểm tra liveness.", attempt = attempt)
        }
        val students = repository.students()
        if (students.isEmpty()) {
            attempt.status = "REJECTED"
            return faceRejected("FACE_PROFILE_NOT_ACTIVE", "Chưa có hồ sơ khuôn mặt để so khớp.", attempt = attempt)
        }
        val ranked = students.map { student ->
            val samples = student.embeddingSamples.ifEmpty { listOf(student.embedding) }
            val score = samples.maxOf { FaceEmbeddingModel.cosine(embedding, it) }
            student to score
        }.sortedByDescending { it.second }
        val (bestStudent, score) = ranked.first()
        val margin = if (ranked.size > 1) score - ranked[1].second else 1f
        val threshold = repository.matchThreshold()
        if (score < threshold || margin < MIN_MARGIN) {
            attempt.status = "REJECTED"
            return FaceResult(
                false,
                "VERIFICATION_FAILED",
                "Không khớp hồ sơ khuôn mặt đã đăng ký.",
                similarity = score,
                margin = margin,
                livenessPassed = true,
                attempt = attempt
            )
        }
        val inserted = repository.markPresent(bestStudent, score, sessionCode)
        attempt.status = "VERIFIED"
        return FaceResult(
            accepted = true,
            code = if (inserted) "PRESENT" else "ALREADY_CHECKED_IN",
            message = if (inserted) "Xác thực thành công." else "Sinh viên đã có mặt trong buổi này.",
            student = bestStudent,
            similarity = score,
            margin = margin,
            attendance = if (inserted) "checked_in" else "already_present",
            livenessPassed = true,
            attempt = attempt
        )
    }

    fun payloadFor(attempt: Attempt?, faceResult: FaceResult? = null): JSONObject = JSONObject().apply {
        put("verificationStage", when {
            attempt?.status == "EXPIRED" -> "EXPIRED"
            attempt?.status == "VERIFIED" -> "VERIFIED"
            attempt?.status == "REJECTED" -> "REJECTED"
            faceResult != null -> if (faceResult.accepted) "VERIFIED" else "REJECTED"
            attempt?.bleEvidence != null -> "BLE_VERIFIED"
            attempt != null -> "SCAN_BLE"
            else -> "IDLE"
        })
        attempt?.let {
            put("attemptId", it.id)
            put("clientAttemptId", it.clientAttemptId)
            put("attemptStatus", it.status)
            it.bleEvidence?.let { evidence ->
                put("bleEvidence", JSONObject()
                    .put("sessionCode", evidence.sessionCode)
                    .put("scanCount", evidence.scanCount)
                    .put("scanDurationMs", evidence.scanDurationMs)
                    .put("rssiMin", evidence.rssiMin)
                    .put("rssiMax", evidence.rssiMax)
                    .put("rssiAverage", evidence.rssiAverage))
            }
            it.challenge?.let { challenge ->
                put("faceChallenge", JSONObject()
                    .put("type", challenge.type)
                    .put("expiresAt", challenge.expiresAt)
                    .put("used", challenge.used))
            }
        }
        faceResult?.let {
            put("verificationCode", it.code)
            put("livenessPassed", it.livenessPassed)
            put("serverSimilarity", it.similarity.toDouble())
            put("serverMargin", it.margin.toDouble())
        }
    }

    private fun issueToken(sessionCode: String): Broadcast {
        require(sessionCode.isNotBlank()) { "sessionCode must not be blank" }
        val token = randomToken(TOKEN_BYTES)
        val issued = Broadcast(
            sessionCode = sessionCode,
            token = token,
            tokenFingerprint = hashToken(token),
            expiresAt = System.currentTimeMillis() + TOKEN_TTL_MS
        )
        broadcasts[sessionCode] = issued
        return issued
    }

    private fun rejected(code: String, message: String, attempt: Attempt? = null) = BleResult(false, code, message, attempt)

    private fun faceRejected(code: String, message: String, attempt: Attempt? = null) = FaceResult(false, code, message, attempt = attempt)

    companion object {
        const val TOKEN_TTL_MS = 90_000L
        const val DEFAULT_ROTATE_MS = 30_000L
        const val ATTEMPT_TTL_MS = 90_000L
        const val FACE_CHALLENGE_TTL_MS = 30_000L
        const val MIN_SCAN_COUNT = 3
        const val MIN_SCAN_DURATION_MS = 2_000L
        const val MIN_RSSI = -85
        const val MIN_MARGIN = 0.035f
        private const val TOKEN_BYTES = 12

        fun hashToken(value: String): String = MessageDigest.getInstance("SHA-256")
            .digest(value.toByteArray(Charsets.UTF_8))
            .joinToString("") { byte -> "%02x".format(byte) }

        private fun randomToken(byteCount: Int): String {
            val bytes = ByteArray(byteCount)
            SecureRandom().nextBytes(bytes)
            return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
        }

        private fun constantTimeEquals(left: String, right: String): Boolean {
            val a = left.toByteArray(Charsets.UTF_8)
            val b = right.toByteArray(Charsets.UTF_8)
            return MessageDigest.isEqual(a, b)
        }
    }
}
