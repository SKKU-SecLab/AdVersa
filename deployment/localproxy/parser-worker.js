const { parentPort } = require('worker_threads');
const { parse } = require('acorn-loose'); // Or your preferred parser

parentPort.on('message', (code) => {
  try {
    const ast = parse(code, { ecmaVersion: 2020 });
    parentPort.postMessage({ ast });
  } catch (e) {
    parentPort.postMessage({ error: e.message });
  }
});