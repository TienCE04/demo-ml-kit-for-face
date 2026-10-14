const SERVICE_UUID = '7c6f1000-8f1b-4d4b-a752-6f656d6f0001';
const RESULT_UUID = '7c6f1001-8f1b-4d4b-a752-6f656d6f0001';
const NOTIFY_UUID = '7c6f1002-8f1b-4d4b-a752-6f656d6f0001';
const $ = (id) => document.getElementById(id);
let device, characteristic, demoTimer, demoTick = 0;
let attended = new Set();
let activeSession = '';

function setConnected(connected, label = connected ? 'Đã kết nối' : 'Chưa kết nối') {
  $('connectionState').classList.toggle('connected', connected);
  $('connectionState').querySelector('span').textContent = label;
  $('streamLabel').textContent = connected ? 'STREAMING' : 'IDLE';
  document.querySelectorAll('.pipeline-node')[2].classList.toggle('active', connected);
  document.querySelectorAll('.pipeline-node')[3].classList.toggle('active', connected);
}

function addEvent(message) {
  const list = $('eventLog');
  list.querySelector('.empty-log')?.remove();
  const item = document.createElement('li');
  const time = new Date().toLocaleTimeString('vi-VN', { hour12: false });
  item.innerHTML = `<time>${time}</time><span></span>`;
  item.lastElementChild.textContent = message;
  list.prepend(item);
  while (list.children.length > 10) list.lastElementChild.remove();
}

function updateView(result, simulated = false) {
  const faces = Number(result.faces ?? result.n ?? 0);
  $('faceCount').textContent = String(faces);
  $('faceCaption').textContent = faces === 1 ? 'Đang theo dõi một khuôn mặt' : faces > 1 ? `Đang theo dõi ${faces} khuôn mặt` : 'Không thấy khuôn mặt';
  const angle = (value) => value == null || Number.isNaN(Number(value)) ? '—' : Math.round(Number(value));
  $('yaw').innerHTML = `${angle(result.yaw)}<small>°</small>`;
  $('pitch').innerHTML = `${angle(result.pitch)}<small>°</small>`;
  if (result.model) $('modelName').textContent = result.source === 'attendance' ? 'ML Kit + MobileFaceNet' : `ML Kit Face Detection · ${result.model}`;
  if (result.source) $('sourceName').textContent = result.source;
  if (result.session) {
    if (activeSession && activeSession !== result.session) {
      attended.clear();
      $('attendanceRows').innerHTML = '<tr><td colspan="4" class="table-empty">Đang tải điểm danh của buổi mới…</td></tr>';
    }
    activeSession = result.session;
    $('sessionName').textContent = result.session;
  }
  if (result.registeredCount != null || result.presentCount != null) {
    $('attendanceCount').innerHTML = `${Number(result.presentCount || 0)}<small> / ${Number(result.registeredCount || 0)}</small>`;
  }
  if (Array.isArray(result.attendees)) result.attendees.forEach(record => addAttendanceRow(record, result.session));
  if (result.attendance) showAttendance(result);
  document.querySelectorAll('.pipeline-node')[2].classList.add('active');
  document.querySelectorAll('.pipeline-node')[3].classList.add('active');
  if (!simulated) addEvent(`${faces} face${faces === 1 ? '' : 's'} · yaw ${angle(result.yaw)}° · pitch ${angle(result.pitch)}°`);
}

function showAttendance(result) {
  if (result.attendance === 'checked_in' && result.studentId) {
    addAttendanceRow(result, result.session);
    addEvent(`Điểm danh · ${result.studentName || result.studentId} · khớp ${Math.round(Number(result.similarity || 0) * 100)}%`);
    $('attendanceStatus').textContent = 'Đã nhận lượt điểm danh';
  } else if (result.attendance === 'already_present') {
    addEvent(`${result.studentName || result.studentId} đã có mặt trong buổi này.`);
  }
}

function addAttendanceRow(record, session) {
  if (!record.studentId) return;
  const key = `${session || ''}:${record.studentId}`;
  if (attended.has(key)) return;
  attended.add(key);
  $('attendanceRows').querySelector('.table-empty')?.closest('tr')?.remove();
  const row = document.createElement('tr');
  const time = new Date(Number(record.updatedAt || Date.now())).toLocaleTimeString('vi-VN', { hour12: false });
  [time, record.studentId, record.studentName || 'Sinh viên', `${Math.round(Number(record.similarity || 0) * 100)}%`].forEach(value => {
    const cell = document.createElement('td'); cell.textContent = value; row.appendChild(cell);
  });
  $('attendanceRows').prepend(row);
  $('attendanceStatus').textContent = 'Đã đồng bộ roster điểm danh';
}

function decode(event) {
  const bytes = new TextDecoder().decode(event.target.value);
  try { updateView(JSON.parse(bytes)); }
  catch { addEvent(`Không đọc được payload: ${bytes.slice(0, 45)}`); }
}

async function connect() {
  if (!navigator.bluetooth) {
    $('compatibility').textContent = 'Trình duyệt này chưa hỗ trợ Web Bluetooth. Dùng Chrome/Edge bản desktop hoặc Android, hoặc bật mô phỏng.';
    addEvent('Web Bluetooth không khả dụng trong trình duyệt hiện tại.');
    return;
  }
  try {
    device = await navigator.bluetooth.requestDevice({ filters: [{ services: [SERVICE_UUID] }] });
    device.addEventListener('gattserverdisconnected', () => {
      characteristic = null;
      setConnected(false);
      addEvent('Thiết bị BLE đã ngắt kết nối.');
    });
    $('connectionState').querySelector('span').textContent = 'Đang kết nối…';
    const server = await device.gatt.connect();
    const service = await server.getPrimaryService(SERVICE_UUID);
    characteristic = await service.getCharacteristic(RESULT_UUID);
    const notifier = await service.getCharacteristic(NOTIFY_UUID);
    await notifier.startNotifications();
    notifier.addEventListener('characteristicvaluechanged', async () => {
      try { updateView(JSON.parse(new TextDecoder().decode(await characteristic.readValue()))); }
      catch (error) { addEvent(`Không đọc được trạng thái BLE: ${error.message}`); }
    });
    setConnected(true, `Đã kết nối · ${device.name || 'Vision Demo'}`);
    updateView(JSON.parse(new TextDecoder().decode(await characteristic.readValue())));
    addEvent('Đã subscribe BLE GATT notify.');
    $('compatibility').textContent = 'Kết nối thành công · kết quả mới sẽ tự cập nhật.';
  } catch (error) {
    $('connectionState').querySelector('span').textContent = 'Chưa kết nối';
    addEvent(error.name === 'NotFoundError' ? 'Không tìm thấy service. Hãy bật BLE trong app Android.' : `BLE: ${error.message}`);
  }
}

function simulate() {
  clearInterval(demoTimer);
  device?.gatt?.connected && device.gatt.disconnect();
  setConnected(true, 'Mô phỏng');
  $('compatibility').textContent = 'Đang hiển thị dữ liệu giả lập. Kết nối thiết bị để xem dữ liệu thật.';
  demoTimer = setInterval(() => {
    demoTick++;
    const visible = demoTick % 9 > 1;
    updateView({ faces: visible ? 1 : 0, yaw: visible ? Math.sin(demoTick / 3) * 24 : null, pitch: visible ? Math.cos(demoTick / 4) * 12 : null, model: 'fast · bundled', source: 'Camera' }, true);
    addEvent(visible ? 'Frame xử lý · 1 face được track' : 'Frame xử lý · không có face');
  }, 1150);
  addEvent('Bắt đầu mô phỏng pipeline.');
}

$('connectBtn').addEventListener('click', () => { clearInterval(demoTimer); connect(); });
$('simulateBtn').addEventListener('click', simulate);
$('clearLog').addEventListener('click', () => { $('eventLog').innerHTML = '<li class="empty-log">Log đã được xóa · chờ sự kiện tiếp theo</li>'; });
if (!navigator.bluetooth) $('compatibility').textContent = 'Web Bluetooth chưa khả dụng ở đây · vẫn có thể xem luồng bằng nút mô phỏng.';
