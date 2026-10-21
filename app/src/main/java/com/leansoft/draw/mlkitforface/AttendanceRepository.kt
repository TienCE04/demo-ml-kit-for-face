package com.leansoft.draw.mlkitforface

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class EnrolledStudent(
    val id: String,
    val name: String,
    val embedding: FloatArray,
    val sampleCount: Int,
    val sampleTypes: List<String> = emptyList(),
    val embeddingSamples: List<FloatArray> = emptyList(),
    val consentVersion: String = "ATTENDANCE_FACE_VERIFICATION_V1",
    val consentAt: Long = System.currentTimeMillis()
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as EnrolledStudent

        if (sampleCount != other.sampleCount) return false
        if (id != other.id) return false
        if (name != other.name) return false
        if (!embedding.contentEquals(other.embedding)) return false
        if (sampleTypes != other.sampleTypes) return false
        if (embeddingSamples.size != other.embeddingSamples.size) return false
        if (embeddingSamples.zip(other.embeddingSamples).any { (left, right) -> !left.contentEquals(right) }) return false
        if (consentVersion != other.consentVersion) return false
        if (consentAt != other.consentAt) return false

        return true
    }

    override fun hashCode(): Int {
        var result = sampleCount
        result = 31 * result + id.hashCode()
        result = 31 * result + name.hashCode()
        result = 31 * result + embedding.contentHashCode()
        result = 31 * result + sampleTypes.hashCode()
        result = 31 * result + embeddingSamples.fold(1) { hash, sample -> 31 * hash + sample.contentHashCode() }
        result = 31 * result + consentVersion.hashCode()
        result = 31 * result + consentAt.hashCode()
        return result
    }
}

data class AttendanceRecord(
    val studentId: String,
    val studentName: String,
    val session: String,
    val timestamp: Long,
    val similarity: Float,
    val status: String = "PRESENT"
)

class AttendanceRepository(context: Context) {
    private val preferences = SecurePreferenceStore(context, "attendance_demo")

    fun currentSession(): String = preferences.getString("session", "${java.time.LocalDate.now()}") ?: "${java.time.LocalDate.now()}"

    fun setSession(session: String) {
        preferences.putString("session", session.trim().ifBlank { "${java.time.LocalDate.now()}" })
    }

    fun matchThreshold(): Float = preferences.getFloat("match_threshold", DEFAULT_MATCH_THRESHOLD)

    fun setMatchThreshold(threshold: Float) {
        require(threshold in 0.1f..0.99f)
        preferences.putFloat("match_threshold", threshold)
    }

    fun students(): List<EnrolledStudent> {
        val array = JSONArray(preferences.getString("students", "[]"))
        return (0 until array.length()).mapNotNull { index ->
            runCatching {
                val item = array.getJSONObject(index)
                val values = item.getJSONArray("embedding")
                val centroid = FloatArray(values.length()) { values.getDouble(it).toFloat() }
                val samples = item.optJSONArray("embeddingSamples")?.let { sampleArray ->
                    (0 until sampleArray.length()).mapNotNull { sampleIndex ->
                        sampleArray.optJSONArray(sampleIndex)?.let { sample ->
                            FloatArray(sample.length()) { valueIndex -> sample.getDouble(valueIndex).toFloat() }
                        }
                    }
                }.orEmpty()
                val types = item.optJSONArray("sampleTypes")?.let { typeArray ->
                    (0 until typeArray.length()).map { typeArray.optString(it) }
                }.orEmpty()
                EnrolledStudent(
                    item.getString("id"),
                    item.getString("name"),
                    centroid,
                    item.optInt("samples", samples.size.coerceAtLeast(1)),
                    types,
                    samples,
                    item.optString("consentVersion", "ATTENDANCE_FACE_VERIFICATION_V1"),
                    item.optLong("consentAt", 0L)
                )
            }.getOrNull()
        }
    }

    fun saveStudent(student: EnrolledStudent) {
        val all = students().filterNot { it.id.equals(student.id, ignoreCase = true) }.toMutableList()
        all += student
        val json = JSONArray()
        all.forEach { entry ->
            val vector = JSONArray()
            entry.embedding.forEach(vector::put)
            val samples = JSONArray()
            entry.embeddingSamples.forEach { sample ->
                samples.put(JSONArray().also { values -> sample.forEach(values::put) })
            }
            val types = JSONArray()
            entry.sampleTypes.forEach(types::put)
            json.put(
                JSONObject()
                    .put("id", entry.id)
                    .put("name", entry.name)
                    .put("samples", entry.sampleCount)
                    .put("embedding", vector)
                    .put("embeddingSamples", samples)
                    .put("sampleTypes", types)
                    .put("consentVersion", entry.consentVersion)
                    .put("consentAt", entry.consentAt)
            )
        }
        preferences.putString("students", json.toString())
    }

    fun removeStudent(id: String) {
        val json = JSONArray()
        students().filterNot { it.id == id }.forEach { entry ->
            val vector = JSONArray()
            entry.embedding.forEach(vector::put)
            val samples = JSONArray()
            entry.embeddingSamples.forEach { sample ->
                samples.put(JSONArray().also { values -> sample.forEach(values::put) })
            }
            val types = JSONArray()
            entry.sampleTypes.forEach(types::put)
            json.put(
                JSONObject()
                    .put("id", entry.id)
                    .put("name", entry.name)
                    .put("samples", entry.sampleCount)
                    .put("embedding", vector)
                    .put("embeddingSamples", samples)
                    .put("sampleTypes", types)
                    .put("consentVersion", entry.consentVersion)
                    .put("consentAt", entry.consentAt)
            )
        }
        preferences.putString("students", json.toString())
        val kept = attendance().filterNot { it.studentId == id }
        saveAttendance(kept)
    }

    fun attendance(): List<AttendanceRecord> {
        val array = JSONArray(preferences.getString("attendance", "[]"))
        return (0 until array.length()).mapNotNull { index ->
            runCatching {
                val item = array.getJSONObject(index)
                AttendanceRecord(
                    item.getString("studentId"),
                    item.getString("studentName"),
                    item.getString("session"),
                    item.getLong("timestamp"),
                    item.getDouble("similarity").toFloat(),
                    item.optString("status", "PRESENT")
                )
            }.getOrNull()
        }.sortedByDescending { it.timestamp }
    }

    fun markPresent(student: EnrolledStudent, score: Float, session: String = currentSession()): Boolean {
        val records = attendance()
        if (records.any { it.studentId == student.id && it.session == session }) return false
        saveAttendance(records + AttendanceRecord(student.id, student.name, session, System.currentTimeMillis(), score, "PRESENT"))
        return true
    }

    fun finalizeSession(session: String = currentSession()): Int {
        val records = attendance()
        val presentIds = records
            .filter { it.session == session && it.status in setOf("PRESENT", "LATE") }
            .map { it.studentId }
            .toSet()
        val now = System.currentTimeMillis()
        val absent = students()
            .filterNot { student -> presentIds.contains(student.id) || records.any { it.session == session && it.studentId == student.id } }
            .map { student -> AttendanceRecord(student.id, student.name, session, now, 0f, "ABSENT") }
        if (absent.isNotEmpty()) saveAttendance(records + absent)
        return absent.size
    }

    fun clearSession(session: String = currentSession()) {
        saveAttendance(attendance().filterNot { it.session == session })
    }

    private fun saveAttendance(records: List<AttendanceRecord>) {
        val json = JSONArray()
        records.forEach { record ->
            json.put(JSONObject().put("studentId", record.studentId).put("studentName", record.studentName).put("session", record.session).put("timestamp", record.timestamp).put("similarity", record.similarity.toDouble()).put("status", record.status))
        }
        preferences.putString("attendance", json.toString())
    }

    companion object {
        const val DEFAULT_MATCH_THRESHOLD = 0.72f
    }
}
