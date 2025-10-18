const { Worker } = require('worker_threads');
const path = require('path');

const TIMEOUT_MS = 3000;
let code = '';

process.stdin.on('data', chunk => {
  code += chunk;
});

process.stdin.on('end', () => {
  const worker = new Worker(path.resolve(__dirname, 'parser-worker.js'));
  
  const timeout = setTimeout(() => {
    console.error(`Parsing timed out after ${TIMEOUT_MS}ms`);
    worker.terminate();
    process.exit(1);
  }, TIMEOUT_MS);

  worker.on('message', (result) => {
    clearTimeout(timeout);

    if (result.error) {
      console.error(`Acorn parsing error: ${result.error}`);
      process.exit(1);
    } else {
      console.log(JSON.stringify(result.ast, null, 0));
      process.exit(0);
    }
  });
  
  worker.on('error', (err) => {
    clearTimeout(timeout);
    console.error(`Worker error: ${err.message}`);
    process.exit(1);
  });
  
  worker.postMessage(code);
});