export const ROWS = 6;
export const COLS = 7;

export type Player = 1 | 2;
export type Cell = Player | null;
export type Board = Cell[][];

export function createEmptyBoard(): Board {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(null));
}

export function dropPiece(board: Board, col: number, player: Player): { board: Board; row: number } | null {
  for (let row = ROWS - 1; row >= 0; row--) {
    if (board[row][col] === null) {
      const newBoard = board.map((r) => [...r]);
      newBoard[row][col] = player;
      return { board: newBoard, row };
    }
  }
  return null;
}

export function checkWinner(board: Board): { player: Player; cells: [number, number][] } | null {
  const directions: [number, number][] = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const player = board[row][col];
      if (!player) continue;

      for (const [dr, dc] of directions) {
        const cells: [number, number][] = [[row, col]];
        for (let i = 1; i < 4; i++) {
          const r = row + dr * i;
          const c = col + dc * i;
          if (r < 0 || r >= ROWS || c < 0 || c >= COLS || board[r][c] !== player) break;
          cells.push([r, c]);
        }
        if (cells.length === 4) return { player, cells };
      }
    }
  }
  return null;
}

export function isBoardFull(board: Board): boolean {
  return board[0].every((cell) => cell !== null);
}

export function isColumnFull(board: Board, col: number): boolean {
  return board[0][col] !== null;
}
