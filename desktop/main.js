/**
 * Electron shell.
 *
 * The app is a Python model behind a local HTTP API plus an HTML front-end.
 * This process owns the lifecycle: pick a free port, start the backend, wait
 * until it answers /health, then show the window. Nothing is ever exposed
 * beyond 127.0.0.1, and the child is killed when the app quits.
 */
const { app, BrowserWindow, shell, dialog, Menu } = require('electron');
const { spawn } = require('node:child_process');
const net = require('node:net');
const path = require('node:path');
const fs = require('node:fs');

const isDev = !app.isPackaged;
const ROOT = isDev ? path.join(__dirname, '..') : process.resourcesPath;

let backend = null;
let win = null;
let backendPort = null;
let starting = null;

/** An OS-assigned free port, so two copies never fight over 8000. */
function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

/**
 * Locate the Python that can run the server.
 * Packaged builds ship a self-contained binary; a dev checkout uses the venv.
 */
function backendCommand(port) {
  const bundled = path.join(ROOT, 'backend', 'predictor-server');
  if (fs.existsSync(bundled)) {
    return { cmd: bundled, args: ['--port', String(port)], cwd: ROOT };
  }
  const venv = path.join(ROOT, '.venv', 'bin', 'python');
  const python = fs.existsSync(venv) ? venv : 'python3';
  return {
    cmd: python,
    args: ['-m', 'uvicorn', 'src.api:app', '--host', '127.0.0.1', '--port', String(port)],
    cwd: ROOT,
  };
}

async function waitForHealth(port, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/health`);
      if (res.ok) return await res.json();
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error('the prediction server did not start in time');
}

async function startBackend() {
  const port = await freePort();
  const { cmd, args, cwd } = backendCommand(port);

  backend = spawn(cmd, args, {
    cwd,
    env: {
      ...process.env,
      // Only a packaged app needs redirecting: its bundle is read-only, so the
      // database and any retrained model must live in Application Support.
      // A dev run must use the checkout, or it would not see the model and data
      // that `make data` and `make train` just produced.
      ...(app.isPackaged
        ? { PREDICTOR_DATA_DIR: path.join(app.getPath('userData'), 'data') }
        : {}),
      PYTHONUNBUFFERED: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  const log = (buf) => process.stdout.write(`[backend] ${buf}`);
  backend.stdout.on('data', log);
  backend.stderr.on('data', log);
  backend.on('exit', (code, signal) => {
    if (code !== 0 && code !== null && !app.isQuitting) {
      dialog.showErrorBox('Prediction server stopped',
        `The backend exited with code ${code}${signal ? ` (${signal})` : ''}.`);
    }
    backend = null;
  });

  await waitForHealth(port);
  backendPort = port;
  return port;
}

function createWindow(port) {
  win = createShell();
  showApp(port);
}

function createShell() {
  const w = new BrowserWindow({
    width: 1120,
    height: 820,
    minWidth: 720,
    minHeight: 560,
    title: 'Match Predictor',
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#0d1117',
    show: false,
    webPreferences: {
      // The renderer only ever talks to the local API over fetch, so it needs
      // no Node access at all.
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });

  // A frozen build takes a while to unpack and import scikit-learn, so put a
  // window on screen straight away rather than leaving a blank dock icon.
  w.loadFile(path.join(__dirname, 'loading.html'));
  w.once('ready-to-show', () => w.show());

  // Anything that is not the local app opens in the real browser.
  w.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
  w.on('closed', () => { win = null; });
  return w;
}

function showApp(port) {
  if (!win) return;
  win.loadURL(`http://127.0.0.1:${port}/`);
  win.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith(`http://127.0.0.1:${port}`)) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });
}

function buildMenu() {
  const template = [
    ...(process.platform === 'darwin' ? [{ role: 'appMenu' }] : []),
    {
      label: 'View',
      submenu: [
        {
          label: 'Reload app',
          accelerator: 'CmdOrCtrl+R',
          click: () => win && win.reload(),
        },
        {
          label: 'Refresh data & retrain if needed',
          accelerator: 'CmdOrCtrl+Shift+R',
          click: async () => {
            if (!backendPort) return;
            await fetch(`http://127.0.0.1:${backendPort}/refresh`, { method: 'POST' })
              .catch(() => {});
            win && win.reload();
          },
        },
        { type: 'separator' },
        { role: 'toggleDevTools' },
        { role: 'togglefullscreen' },
        { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
      ],
    },
    { role: 'windowMenu' },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

app.whenReady().then(async () => {
  buildMenu();
  win = createShell();          // on screen before the backend is even spawned
  try {
    starting = startBackend();
    const port = await starting;
    showApp(port);
  } catch (err) {
    dialog.showErrorBox('Could not start Match Predictor', String(err.message || err));
    app.quit();
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0 && backendPort) createWindow(backendPort);
  });
});

app.on('before-quit', () => { app.isQuitting = true; });

function stopBackend() {
  if (!backend) return;
  backend.kill('SIGTERM');
  // uvicorn normally goes on SIGTERM; make sure it cannot outlive the app.
  const child = backend;
  setTimeout(() => { try { child.kill('SIGKILL'); } catch { /* gone */ } }, 3000);
  backend = null;
}

app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') app.quit();
});
app.on('quit', stopBackend);
process.on('exit', stopBackend);
