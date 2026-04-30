import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("연결 시도 중...");
  const [logs, setLogs] = useState([]);
  const socket = useRef(null);

  useEffect(() => {
    const connect = () => {
      socket.current = new WebSocket("ws://localhost:8000/ws");

      socket.current.onopen = () => {
        setStatus("연결 성공 (관제 중)");
        console.log("Connected to Backend");
      };

      socket.current.onmessage = (event) => {
        const response = JSON.parse(event.data);
        if (response.type === "data") {
          setData(response);
          setLogs(prev => [response, ...prev].slice(0, 15));
        } else if (response.type === "status") {
          setStatus(response.message === "recording" ? "음성 분석 중..." : "일시 정지");
        }
      };

      socket.current.onclose = () => {
        setStatus("서버 연결 끊김. 재시도 중...");
        setTimeout(connect, 3000); // 3초 후 재연결
      };

      socket.current.onerror = (err) => {
        console.error("Socket Error: ", err);
        socket.current.close();
      };
    };

    connect();
    return () => socket.current?.close();
  }, []);

  const sendCommand = (action, key = "") => {
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify({ type: "CONTROL", action, key }));
    }
  };

  return (
    <div className="dashboard">
      <header className="header">
        <h1>🛡️ SoundGuard Control Center</h1>
        <div style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
          <span>{status}</span>
          <button className="btn-pause" onClick={() => sendCommand("PAUSE")}>⏸ 감지 일시정지</button>
        </div>
      </header>

      {/* 왼쪽: 상태 및 분석 */}
      <aside className="card-group">
        <div className={`card ${data?.status.level > 0 ? 'status-critical' : ''}`}>
          <small>현재 상태</small>
          <h2>{data?.status.name || "정상상황"}</h2>
          <div className="timer">{data?.status.duration.toString().padStart(2, '0')}:00</div>
        </div>

        <div className="card" style={{marginTop: '20px'}}>
          <h3>음량 분석</h3>
          {['footstep', 'voice', 'noise'].map(key => (
            <div key={key} className="progress-container">
              <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.7rem'}}>
                <span>{key.toUpperCase()}</span>
                <span>{data?.analysis.scores[key] || 0}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{width: `${data?.analysis.scores[key] || 0}%`}}></div>
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* 중앙: 지도 및 실시간 알림 */}
      <main>
        <div className="card" style={{height: '400px', position: 'relative', display:'flex', justifyContent:'center', alignItems:'center'}}>
          <div style={{textAlign:'center', color:'#475569'}}>
            <div className="animate-ping" style={{width:'20px', height:'200px', position:'absolute'}}></div>
            📍 수락산 위험구간 감시 중<br/>
            <small>37.5665° N, 126.9780° E</small>
          </div>
        </div>
        <div className="card" style={{marginTop: '20px', borderLeft: '5px solid #3b82f6'}}>
          <span style={{color: '#3b82f6', fontWeight: 'bold'}}>📢 안내 메시지:</span>
          <p>{data?.action_msg || "시스템 가동 중입니다."}</p>
        </div>
      </main>

      {/* 오른쪽: 제어 및 로그 */}
      <aside className="card-group">
        <div className="card btn-group">
          <h3>강제 방송 제어</h3>
          <button className="btn-primary" onClick={() => sendCommand("FORCE_TTS", "INTRUSION_WARN_1")}>1차 경고 방송</button>
          <button onClick={() => sendCommand("FORCE_TTS", "EMERGENCY_GUIDE")}>응급 구조 안내</button>
        </div>

        <div className="card" style={{marginTop: '20px', height: '350px', overflowY: 'auto'}}>
          <h3>이벤트 로그</h3>
          {logs.map((log, i) => (
            <div key={i} className="log-item">
              <div style={{color:'#64748b'}}>{log.timestamp}</div>
              <strong>{log.status.name}</strong>
              <div style={{fontSize:'0.7rem'}}>{log.stt_text || log.analysis.label}</div>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}

export default App;