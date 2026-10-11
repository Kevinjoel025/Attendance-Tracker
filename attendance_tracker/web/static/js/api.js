/**
 * attendance_tracker/web/static/js/api.js
 * ───────────────────────────────────────
 * Clean fetch API client for Attendance Assistant & Backend Services.
 */

const API = {
  /**
   * Submit natural language query to the SLM NLU & ToolRouter.
   */
  async askAssistant(query) {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query.trim() })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Assistant request failed: ${res.statusText}`);
    }
    return await res.json();
  },

  /**
   * Fetch complete dashboard overview metrics and subject list.
   */
  async getDashboard() {
    const res = await fetch('/api/dashboard');
    if (!res.ok) throw new Error(`Failed to load dashboard: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch subject list.
   */
  async getSubjects() {
    const res = await fetch('/api/subjects');
    if (!res.ok) throw new Error(`Failed to load subjects: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch full subject analytics.
   */
  async getSubjectDetail(subjectId) {
    const res = await fetch(`/api/subjects/${encodeURIComponent(subjectId)}`);
    if (!res.ok) throw new Error(`Subject not found: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Trigger real portal synchronization.
   * Checks both HTTP status and response success flag.
   */
  async syncAttendance() {
    const res = await fetch('/api/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.success === false) {
      throw new Error(data.error_message || data.message || data.detail || `Sync failed: ${res.statusText}`);
    }
    return data;
  },

  /**
   * Get weekly timetable schedule.
   */
  async getWeeklyTimetable() {
    const res = await fetch('/api/timetable/weekly');
    if (!res.ok) throw new Error(`Failed to load weekly timetable: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Get calendar day status.
   */
  async getCalendarStatus(dateStr) {
    const url = dateStr ? `/api/calendar/status?date=${encodeURIComponent(dateStr)}` : '/api/calendar/status';
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to load calendar status: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Get unified authoritative list of all calendar entries.
   */
  async getUnifiedCalendarEntries() {
    const res = await fetch('/api/calendar/entries');
    if (!res.ok) throw new Error(`Failed to load calendar entries: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Add or edit a calendar entry/override.
   */
  async saveCalendarEntry(entryData) {
    const res = await fetch('/api/calendar/entries', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(entryData)
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to save calendar entry');
    return data;
  },

  /**
   * Delete a calendar entry or remove override for date.
   */
  async deleteCalendarEntry(dateStr) {
    const res = await fetch(`/api/calendar/entries/${encodeURIComponent(dateStr)}`, {
      method: 'DELETE'
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to delete calendar entry');
    return data;
  },

  /**
   * Get Mid-Term 1 configured start and end dates.
   */
  async getMidTerm1() {
    const res = await fetch('/api/calendar/midterm');
    if (!res.ok) throw new Error(`Failed to load Mid-Term 1: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Update Mid-Term 1 start and end dates.
   */
  async saveMidTerm1(start, end) {
    const res = await fetch('/api/calendar/midterm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ start, end })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to save Mid-Term 1 dates');
    return data;
  },

  /**
   * List calendar overrides.
   */
  async getCalendarOverrides() {
    const res = await fetch('/api/calendar/overrides');
    if (!res.ok) throw new Error(`Failed to load calendar overrides: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Add or update calendar override.
   */
  async addCalendarOverride(date, override_type, reason = '') {
    const res = await fetch('/api/calendar/overrides', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ date, override_type, reason })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to save override');
    return data;
  },

  /**
   * Delete calendar override for date.
   */
  async deleteCalendarOverride(dateStr) {
    const res = await fetch(`/api/calendar/overrides/${encodeURIComponent(dateStr)}`, {
      method: 'DELETE'
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to delete override');
    return data;
  },

  /**
   * Get semester last working day.
   */
  async getLastWorkingDay() {
    const res = await fetch('/api/calendar/last-working-day');
    if (!res.ok) throw new Error(`Failed to load last working day: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Set semester last working day.
   */
  async setLastWorkingDay(last_working_day) {
    const res = await fetch('/api/calendar/last-working-day', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ last_working_day })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to update last working day');
    return data;
  },

  /**
   * List semester official holidays.
   */
  async getSemesterHolidays() {
    const res = await fetch('/api/calendar/holidays');
    if (!res.ok) throw new Error(`Failed to load holidays: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Preview CSV attendance text.
   */
  async previewCsvText(csvText) {
    const res = await fetch('/api/import/preview-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ csv_text: csvText })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to parse CSV');
    return data;
  },

  /**
   * Upload and preview CSV or Excel attendance file.
   */
  async uploadAttendanceFile(formData) {
    const res = await fetch('/api/import/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to parse file');
    return data;
  },

  /**
   * Commit imported attendance records.
   */
  async commitImport(records, source = 'manual') {
    const res = await fetch('/api/import/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records, source })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to commit import');
    return data;
  },

  /**
   * Preview CSV holiday/calendar text.
   */
  async previewCalendarCsvText(csvText) {
    const res = await fetch('/api/import/calendar/preview-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ csv_text: csvText })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to parse holiday CSV');
    return data;
  },

  /**
   * Upload and preview CSV or Excel holiday/calendar file.
   */
  async uploadCalendarFile(formData) {
    const res = await fetch('/api/import/calendar/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to parse holiday file');
    return data;
  },

  /**
   * Commit imported holiday/calendar records.
   */
  async commitCalendarImport(records, source = 'csv') {
    const res = await fetch('/api/import/calendar/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records, source })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Failed to commit holiday import');
    return data;
  },

  /**
   * List snapshot history.
   */
  async getSnapshots() {
    const res = await fetch('/api/snapshots');
    if (!res.ok) throw new Error(`Failed to load snapshots: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Compare snapshots.
   */
  async compareSnapshots(currId = null, prevId = null) {
    let url = '/api/snapshots/compare';
    const params = [];
    if (currId) params.push(`curr_id=${encodeURIComponent(currId)}`);
    if (prevId) params.push(`prev_id=${encodeURIComponent(prevId)}`);
    if (params.length > 0) url += `?${params.join('&')}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to compare snapshots: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch recent change history and schedule.
   */
  async getUpdates() {
    const res = await fetch('/api/updates');
    if (!res.ok) throw new Error(`Failed to load updates: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Check backend health.
   */
  async getHealth() {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
  }
};

window.API = API;
