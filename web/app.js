const GRID_SIZE = 28;
const CELL_COUNT = GRID_SIZE * GRID_SIZE;
const pixels = new Array(CELL_COUNT).fill(0);

const grid = document.querySelector("#grid");
const clearButton = document.querySelector("#clearButton");
const predictButton = document.querySelector("#predictButton");
const modelInfo = document.querySelector("#modelInfo");
const resultTitle = document.querySelector("#resultTitle");
const winnerDigit = document.querySelector("#winnerDigit");
const winnerConfidence = document.querySelector("#winnerConfidence");
const bars = document.querySelector("#bars");

let isDrawing = false;
const cells = [];

function createGrid() {
  for (let index = 0; index < CELL_COUNT; index += 1) {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.dataset.index = index;
    grid.appendChild(cell);
    cells.push(cell);
  }
}

function createBars() {
  for (let digit = 0; digit < 10; digit += 1) {
    const card = document.createElement("div");
    card.className = "bar-card";
    card.innerHTML = `
      <div class="bar-track"><div class="bar-fill"></div></div>
      <div class="digit-label">${digit}</div>
      <div class="percent-label">0.0%</div>
    `;
    bars.appendChild(card);
  }
}

function paintCell(index, amount = 0.9) {
  if (index < 0 || index >= CELL_COUNT) {
    return;
  }

  const row = Math.floor(index / GRID_SIZE);
  const col = index % GRID_SIZE;
  const brush = [
    [row, col, amount],
    [row - 1, col, amount * 0.42],
    [row + 1, col, amount * 0.42],
    [row, col - 1, amount * 0.42],
    [row, col + 1, amount * 0.42],
  ];

  for (const [brushRow, brushCol, value] of brush) {
    if (brushRow < 0 || brushRow >= GRID_SIZE || brushCol < 0 || brushCol >= GRID_SIZE) {
      continue;
    }
    const brushIndex = brushRow * GRID_SIZE + brushCol;
    pixels[brushIndex] = Math.min(1, pixels[brushIndex] + value);
    const lightness = Math.round(pixels[brushIndex] * 255);
    cells[brushIndex].style.backgroundColor = `rgb(${lightness}, ${lightness}, ${lightness})`;
  }
}

function cellFromPointer(event) {
  const target = document.elementFromPoint(event.clientX, event.clientY);
  if (!target || !target.classList.contains("cell")) {
    return null;
  }
  return Number(target.dataset.index);
}

function clearGrid() {
  pixels.fill(0);
  for (const cell of cells) {
    cell.style.backgroundColor = "rgb(0, 0, 0)";
  }
  resultTitle.textContent = "等待输入数字";
  winnerDigit.textContent = "?";
  winnerConfidence.textContent = "置信度将在这里显示";
  updateBars(new Array(10).fill(0), -1);
}

function updateBars(probabilities, prediction) {
  const cards = [...bars.querySelectorAll(".bar-card")];
  probabilities.forEach((probability, digit) => {
    const card = cards[digit];
    const fill = card.querySelector(".bar-fill");
    const label = card.querySelector(".percent-label");
    card.classList.remove("winner");
    fill.style.height = `${Math.max(0, Math.min(1, probability)) * 100}%`;
    label.textContent = `${(probability * 100).toFixed(1)}%`;
  });
}

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "无法加载模型");
    }
  } catch (error) {
    console.error(`模型未就绪：${error.message}`);
  }
}

async function predictDigit() {
  predictButton.disabled = true;
  resultTitle.textContent = "正在识别...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "预测失败");
    }

    updateBars(data.probabilities, data.prediction);
    resultTitle.textContent = `模型认为是 ${data.prediction}`;
    winnerDigit.textContent = data.prediction;
    winnerConfidence.textContent = `置信度 ${(data.confidence * 100).toFixed(2)}%`;
  } catch (error) {
    resultTitle.textContent = "预测失败";
    winnerConfidence.textContent = error.message;
  } finally {
    predictButton.disabled = false;
  }
}

grid.addEventListener("pointerdown", (event) => {
  isDrawing = true;
  grid.setPointerCapture(event.pointerId);
  const index = cellFromPointer(event);
  if (index !== null) {
    paintCell(index);
  }
});

grid.addEventListener("pointermove", (event) => {
  if (!isDrawing) {
    return;
  }
  const index = cellFromPointer(event);
  if (index !== null) {
    paintCell(index);
  }
});

grid.addEventListener("pointerup", () => {
  isDrawing = false;
});

grid.addEventListener("pointerleave", () => {
  isDrawing = false;
});

clearButton.addEventListener("click", clearGrid);
predictButton.addEventListener("click", predictDigit);

createGrid();
createBars();
clearGrid();
loadModelInfo();
