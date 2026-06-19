const fs = require('fs');
const path = require('path');

const projectRoot = path.resolve(__dirname, '..');
const pathsToRemove = ['.next', 'out'];

for (const relativePath of pathsToRemove) {
  const absolutePath = path.join(projectRoot, relativePath);

  if (fs.existsSync(absolutePath)) {
    fs.rmSync(absolutePath, { recursive: true, force: true });
    console.log(`Removed ${relativePath}`);
  }
}
