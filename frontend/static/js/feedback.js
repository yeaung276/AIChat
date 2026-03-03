const sessionId = new URLSearchParams(window.location.search).get('session_id');

document.getElementById('feedback-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const form = e.target;
    const submitBtn = document.getElementById('submit-btn');
    const errorEl = document.getElementById('error-message');
    const loadingEl = document.getElementById('loading');

    errorEl.hidden = true;
    submitBtn.disabled = true;
    loadingEl.hidden = false;

    const data = {
        session_id: parseInt(sessionId),
        q1: parseInt(form.q1.value),
        q2: parseInt(form.q2.value),
        q3: parseInt(form.q3.value),
        q4: parseInt(form.q4.value),
        q5: form.q5.value.trim() || null,
    };

    try {
        const res = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Something went wrong. Please try again.');
        }

        document.getElementById('success-screen').hidden = false;
    } catch (err) {
        errorEl.textContent = err.message;
        errorEl.hidden = false;
        submitBtn.disabled = false;
    } finally {
        loadingEl.hidden = true;
    }
});
