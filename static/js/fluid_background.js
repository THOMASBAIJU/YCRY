const canvas = document.getElementById('bg-canvas');
const ctx = canvas.getContext('2d');

let width, height;
let particles = [];

// Configuration
const PARTICLE_COUNT = 60;
const CONNECTION_DISTANCE = 150;
const MOUSE_RADIUS = 250;
const PARTICLE_SPEED = 0.5;

// Resize handling
function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    initParticles();
}

const ICONS = ['🧸', '🍼', '🦆', '⭐️', '👶', '🎀'];

class Particle {
    constructor() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * PARTICLE_SPEED;
        this.vy = (Math.random() - 0.5) * PARTICLE_SPEED;
        this.size = Math.random() * 15 + 10; // Larger size for emojis
        this.text = ICONS[Math.floor(Math.random() * ICONS.length)];
        // Opacity for soft background feel
        this.opacity = Math.random() * 0.3 + 0.1;
    }

    update(mouseX, mouseY) {
        // Normal movement
        this.x += this.vx;
        this.y += this.vy;

        // Bounce off edges
        if (this.x < 0 || this.x > width) this.vx *= -1;
        if (this.y < 0 || this.y > height) this.vy *= -1;

        // Mouse Interaction (Antigravity/Repulsion)
        if (mouseX != null) {
            let dx = mouseX - this.x;
            let dy = mouseY - this.y;
            let distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < MOUSE_RADIUS) {
                const forceDirectionX = dx / distance;
                const forceDirectionY = dy / distance;
                const force = (MOUSE_RADIUS - distance) / MOUSE_RADIUS;
                const directionX = forceDirectionX * force * 2; // Repulsion strength
                const directionY = forceDirectionY * force * 2;

                this.x -= directionX;
                this.y -= directionY;
            }
        }
    }

    draw() {
        ctx.font = `${this.size}px Arial`;
        ctx.globalAlpha = this.opacity;
        ctx.fillText(this.text, this.x, this.y);
        ctx.globalAlpha = 1.0; // Reset
    }
}

function initParticles() {
    particles = [];
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push(new Particle());
    }
}

// Animation Loop
let mouseX = null;
let mouseY = null;

function animate() {
    ctx.clearRect(0, 0, width, height);

    // Update and draw particles
    particles.forEach(p => {
        p.update(mouseX, mouseY);
        p.draw();
    });

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
        for (let j = i; j < particles.length; j++) {
            let dx = particles[i].x - particles[j].x;
            let dy = particles[i].y - particles[j].y;
            let distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < CONNECTION_DISTANCE) {
                ctx.beginPath();
                ctx.strokeStyle = `rgba(13, 148, 136, ${(1 - distance / CONNECTION_DISTANCE) * 0.2})`;
                ctx.lineWidth = 1;
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.stroke();
            }
        }
    }

    requestAnimationFrame(animate);
}

// Event Listeners
window.addEventListener('resize', resize);
window.addEventListener('mousemove', e => {
    mouseX = e.x;
    mouseY = e.y;
});
window.addEventListener('mouseleave', () => {
    mouseX = null;
    mouseY = null;
});

// Start
resize();
animate();
