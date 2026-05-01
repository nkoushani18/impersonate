import { useState, useEffect, useRef } from 'react'
import { Brain, Send, Trash, Info, User, Sparkles, X } from './Icons'

// Vibrant persona themes
const PERSONA_THEMES = {
    0: { accent1: '#667eea', accent2: '#764ba2' }, // Indigo/Purple
    1: { accent1: '#f093fb', accent2: '#f5576c' }, // Pink/Red
    2: { accent1: '#4facfe', accent2: '#00f2fe' }, // Blue/Cyan
    3: { accent1: '#43e97b', accent2: '#38f9d7' }, // Green/Teal
    4: { accent1: '#fa709a', accent2: '#fee140' }, // Pink/Yellow
}

function App() {
    const [personas, setPersonas] = useState([])
    const [activePersona, setActivePersona] = useState(null)
    const [chatHistories, setChatHistories] = useState({})
    const [inputValue, setInputValue] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [isTyping, setIsTyping] = useState(false)
    const [showThinking, setShowThinking] = useState(false)

    const messagesEndRef = useRef(null)
    const inputRef = useRef(null)

    const currentMessages = activePersona ? (chatHistories[activePersona.persona_id] || []) : []

    // Update theme colors based on active persona
    useEffect(() => {
        if (activePersona) {
            const index = activePersona.persona_id.charCodeAt(activePersona.persona_id.length - 1) % 5
            const theme = PERSONA_THEMES[index]
            document.documentElement.style.setProperty('--accent-1', theme.accent1)
            document.documentElement.style.setProperty('--accent-2', theme.accent2)
        }
    }, [activePersona])

    useEffect(() => {
        fetchPersonas()
    }, [])

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [currentMessages, isTyping])

    useEffect(() => {
        if (activePersona) inputRef.current?.focus()
    }, [activePersona])

    const fetchPersonas = async () => {
        try {
            const res = await fetch('/api/personas')
            const data = await res.json()
            setPersonas(data.personas || [])
        } catch (err) {
            console.error('Failed to fetch personas:', err)
        }
    }

    const selectPersona = async (id) => {
        try {
            const res = await fetch('/api/persona/select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ persona_id: id })
            })
            const data = await res.json()
            if (data.success) {
                setActivePersona(data.persona)
                // Save to localStorage for auto-select in other interfaces
                localStorage.setItem('activePersonaId', id)
                if (!chatHistories[id]) {
                    setChatHistories(prev => ({ ...prev, [id]: [] }))
                }
            }
        } catch (err) {
            console.error('Error selecting persona:', err)
        }
    }

    const sendMessage = async () => {
        if (!inputValue.trim() || !activePersona || isLoading) return

        const text = inputValue.trim()
        setInputValue('')
        setIsLoading(true)
        setIsTyping(true)

        const userMsg = {
            id: Date.now(),
            type: 'user',
            text,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }

        setChatHistories(prev => ({
            ...prev,
            [activePersona.persona_id]: [...(prev[activePersona.persona_id] || []), userMsg]
        }))

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, debug: showThinking })
            })
            const data = await res.json()

            // Simulate natural typing delay
            await new Promise(r => setTimeout(r, 600 + Math.random() * 400))

            const aiMsg = {
                id: Date.now() + 1,
                type: 'persona',
                text: data.response,
                debug: data.debug,
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }

            setChatHistories(prev => ({
                ...prev,
                [activePersona.persona_id]: [...(prev[activePersona.persona_id] || []), aiMsg]
            }))
        } catch (err) {
            console.error('Chat error:', err)
        } finally {
            setIsLoading(false)
            setIsTyping(false)
        }
    }

    const clearChat = () => {
        if (activePersona) {
            setChatHistories(prev => ({ ...prev, [activePersona.persona_id]: [] }))
        }
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    const getInitials = (name) => {
        if (!name) return '??'
        return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    }

    return (
        <div className="app">
            {/* Animated Background */}
            <div className="bg-gradient"></div>

            {/* Sidebar */}
            <aside className="sidebar">
                <div className="sidebar-header">
                    <div className="logo">
                        <div className="logo-icon">
                            <Brain className="w-5 h-5" style={{ color: 'white' }} />
                        </div>
                        <span className="logo-text">PentaAI</span>
                    </div>
                </div>

                <div className="personas-section">
                    <div className="section-label">Personas</div>
                    <div className="persona-list">
                        {personas.map((p, idx) => (
                            <div
                                key={p.id}
                                className={`persona-item ${activePersona?.persona_id === p.id ? 'active' : ''}`}
                                onClick={() => selectPersona(p.id)}
                            >
                                <div className="persona-avatar">
                                    {getInitials(p.name)}
                                </div>
                                <div className="persona-details">
                                    <div className="persona-name">{p.name}</div>
                                    <div className="persona-status">
                                        {Object.keys(p.traits || {}).length} traits loaded
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="sidebar-footer">
                    <button
                        className={`btn ${showThinking ? 'active' : ''}`}
                        onClick={() => setShowThinking(!showThinking)}
                    >
                        <Sparkles className="w-4 h-4" />
                        {showThinking ? 'Thinking ON' : 'Show Thinking'}
                    </button>
                    <button className="btn btn-icon" onClick={clearChat} title="Clear chat">
                        <Trash className="w-4 h-4" />
                    </button>
                </div>
            </aside>

            {/* Chat Area */}
            <main className="chat-area">
                {activePersona ? (
                    <>
                        {/* Header */}
                        <header className="chat-header">
                            <div className="active-persona-avatar">
                                {getInitials(activePersona.name)}
                            </div>
                            <div className="active-persona-info">
                                <h2>{activePersona.name}</h2>
                                <div className="online-status">
                                    <span className="online-dot"></span>
                                    Active now
                                </div>
                            </div>
                            <div className="header-actions">
                                <button className="icon-btn" title="Info">
                                    <Info className="w-4 h-4" />
                                </button>
                            </div>
                        </header>

                        {/* Messages */}
                        <div className="messages-wrapper">
                            {currentMessages.length === 0 ? (
                                <div className="empty-state">
                                    <div className="empty-icon">
                                        <Brain className="w-10 h-10" style={{ color: 'white' }} />
                                    </div>
                                    <h2>Start chatting with {activePersona.name}</h2>
                                    <p>Ask about life decisions, career choices, or relationships. Responses are based on {activePersona.name}'s unique personality traits.</p>
                                    <div className="suggestion-chips">
                                        <button className="chip" onClick={() => { setInputValue("Career or family?"); inputRef.current?.focus() }}>
                                            Career or family?
                                        </button>
                                        <button className="chip" onClick={() => { setInputValue("What matters most to you?"); inputRef.current?.focus() }}>
                                            What matters most?
                                        </button>
                                        <button className="chip" onClick={() => { setInputValue("How do you handle stress?"); inputRef.current?.focus() }}>
                                            How do you handle stress?
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <>
                                    {currentMessages.map(msg => (
                                        <div key={msg.id} className={`message ${msg.type}`}>
                                            <div className="message-avatar">
                                                {msg.type === 'user'
                                                    ? <User className="w-4 h-4" />
                                                    : <Brain className="w-4 h-4" style={{ color: 'white' }} />
                                                }
                                            </div>
                                            <div className="message-content">
                                                <div className="bubble">{msg.text}</div>
                                                <span className="message-time">{msg.time}</span>

                                                {msg.debug && showThinking && (
                                                    <div className="debug-box">
                                                        <div className="debug-label">Reasoning</div>
                                                        <div>Intent: {msg.debug.intent} ({Math.round(msg.debug.confidence * 100)}%)</div>
                                                        <div className="trait-tags">
                                                            {Object.entries(msg.debug.traits_used || {}).map(([k, v]) => (
                                                                <span key={k} className="trait-tag">{k}: {v}</span>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}

                                    {isTyping && (
                                        <div className="message persona">
                                            <div className="message-avatar">
                                                <Brain className="w-4 h-4" style={{ color: 'white' }} />
                                            </div>
                                            <div className="typing-indicator">
                                                <span className="typing-dot"></span>
                                                <span className="typing-dot"></span>
                                                <span className="typing-dot"></span>
                                            </div>
                                        </div>
                                    )}
                                    <div ref={messagesEndRef} />
                                </>
                            )}
                        </div>

                        {/* Input */}
                        <div className="input-area">
                            <div className="input-container">
                                <input
                                    ref={inputRef}
                                    type="text"
                                    className="text-input"
                                    placeholder={`Message ${activePersona.name}...`}
                                    value={inputValue}
                                    onChange={e => setInputValue(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    disabled={isLoading}
                                />
                                <button
                                    className="send-btn"
                                    onClick={sendMessage}
                                    disabled={!inputValue.trim() || isLoading}
                                >
                                    <Send className="w-5 h-5" />
                                </button>
                            </div>
                        </div>
                    </>
                ) : (
                    <div className="empty-state">
                        <div className="empty-icon">
                            <Brain className="w-10 h-10" style={{ color: 'white' }} />
                        </div>
                        <h2>Welcome to PentaAI</h2>
                        <p>Select a persona from the sidebar to start a conversation. Each persona has unique traits that shape their responses.</p>
                    </div>
                )}
            </main>
        </div>
    )
}

export default App
