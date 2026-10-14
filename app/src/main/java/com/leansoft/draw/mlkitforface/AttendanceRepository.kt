package com.leansoft.draw.mlkitforface

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class EnrolledStudent(
    val id: String,
    val name: String,
    val embedding: FloatArray,
    val sampleCount: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as EnrolledStudent

        if (sampleCount != other.sampleCount) return false
        if (id != other.id) return false
        if (name != other.name) return false
        if (!embedding.contentEquals(other.embedding)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = sampleCount
        result = 31 * result + id.hashCode()
        result = 31 * result + name.hashCode()
        result = 31 * result + embedding.contentHashCode()
        return result
    }
}

data class AttendanceRecord(
    val studentId: String,
    val studentName: String,
    val session: String,
    val timestamp: Long,
    val similarity: Float
)

class AttendanceRepository(context: Context) {
    private val preferences = context.getSharedPreferences("attendance_demo", Context.MODE_PRIVATE)

    fun currentSession(): String = preferences.getString("session", "${java.time.LocalDate.now()}") ?: "${java.time.LocalDate.now()}"

    fun setSession(session: String) {
        preferences.edit().putString("session", session.trim().ifBlank { "${java.time.LocalDate.now()}" }).apply()
    }

    fun matchThreshold(): Float = preferences.getFloat("match_threshold", DEFAULT_MATCH_THRESHOLD)

    fun setMatchThreshold(threshold: Float) {
        require(threshold in 0.1f..0.99f)
        preferences.edit().putFloat("match_threshold", threshold).apply()
    }

    fun students(): List<EnrolledStudent> {
        val array = JSONArray(preferences.getString("students", "[]"))
        return (0 until array.length()).mapNotNull { index ->
            runCatching {
                val item = array.getJSONObject(index)
                val values = item.getJSONArray("embedding")
                EnrolledStudent(item.getString("id"), item.getString("name"), FloatArray(values.length()) { values.getDouble(it).toFloat() }, item.optInt("samples", 1))
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
            json.put(JSONObject().put("id", entry.id).put("name", entry.name).put("samples", entry.sampleCount).put("embedding", vector))
        }
        preferences.edit().putString("students", json.toString()).apply()
    }

    fun removeStudent(id: String) {
        val json = JSONArray()
        students().filterNot { it.id == id }.forEach { entry ->
            val vector = JSONArray()
            entry.embedding.forEach(vector::put)
            json.put(JSONObject().put("id", entry.id).put("name", entry.name).put("samples", entry.sampleCount).put("embedding", vector))
        }
        preferences.edit().putString("students", json.toString()).apply()
        val kept = attendance().filterNot { it.studentId == id }
        saveAttendance(kept)
    }

    fun attendance(): List<AttendanceRecord> {
        val array = JSONArray(preferences.getString("attendance", "[]"))
        return (0 until array.length()).mapNotNull { index ->
            runCatching {
                val item = array.getJSONObject(index)
                AttendanceRecord(item.getString("studentId"), item.getString("studentName"), item.getString("session"), item.getLong("timestamp"), item.getDouble("similarity").toFloat())
            }.getOrNull()
        }.sortedByDescending { it.timestamp }
    }

    fun markPresent(student: EnrolledStudent, score: Float, session: String = currentSession()): Boolean {
        val records = attendance()
        if (records.any { it.studentId == student.id && it.session == session }) return false
        saveAttendance(records + AttendanceRecord(student.id, student.name, session, System.currentTimeMillis(), score))
        return true
    }

    fun clearSession(session: String = currentSession()) {
        saveAttendance(attendance().filterNot { it.session == session })
    }

    private fun saveAttendance(records: List<AttendanceRecord>) {
        val json = JSONArray()
        records.forEach { record ->
            json.put(JSONObject().put("studentId", record.studentId).put("studentName", record.studentName).put("session", record.session).put("timestamp", record.timestamp).put("similarity", record.similarity.toDouble()))
        }
        preferences.edit().putString("attendance", json.toString()).apply()
    }

    companion object {
        const val DEFAULT_MATCH_THRESHOLD = 0.72f
    }
}
