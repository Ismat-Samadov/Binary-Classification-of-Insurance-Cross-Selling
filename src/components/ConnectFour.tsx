"use client";

import { useState, useCallback } from "react";
import {
  Board,
  Player,
  ROWS,
  COLS,
  createEmptyBoard,
  dropPiece,
  checkWinner,
  isBoardFull,
  isColumnFull,
} from "@/lib/gameLogic";

type GameState = "playing" | "won" | "draw";

const PLAYER_COLORS: Record<Player, string> = {
  1: "bg-red-500",
  2: "bg-yellow-400",
};

const PLAYER_SHADOW: Record<Player, string> = {
  1: "shadow-red-500/60",
  2: "shadow-yellow-400/60",
};

const PLAYER_RING: Record<Player, string> = {
  1: "ring-red-400",
  2: "ring-yellow-300",
};

const PLAYER_BORDER: Record<Player, string> = {
  1: "border-red-400",
  2: "border-yellow-300",
};

const PLAYER_TEXT: Record<Player, string> = {
  1: "text-red-400",
  2: "text-yellow-300",
};

const PLAYER_NAMES: Record<Player, string> = {
  1: "Red",
  2: "Yellow",
};

export default function ConnectFour() {
  const [board, setBoard] = useState<Board>(createEmptyBoard());
  const [currentPlayer, setCurrentPlayer] = useState<Player>(1);
  const [gameState, setGameState] = useState<GameState>("playing");
  const [winningCells, setWinningCells] = useState<Set<string>>(new Set());
  const [scores, setScores] = useState<Record<Player, number>>({ 1: 0, 2: 0 });
  const [hoveredCol, setHoveredCol] = useState<number | null>(null);
  const [animatingCells, setAnimatingCells] = useState<Set<string>>(new Set());

  const cellKey = (row: number, col: number) => `${row}-${col}`;

  const handleColumnClick = useCallback(
    (col: number) => {
      if (gameState !== "playing") return;
      if (isColumnFull(board, col)) return;

      const result = dropPiece(board, col, currentPlayer);
      if (!result) return;

      const { board: newBoard, row } = result;
      const key = cellKey(row, col);

      setAnimatingCells((prev) => new Set(prev).add(key));
      setTimeout(() => {
        setAnimatingCells((prev) => {
          const next = new Set(prev);
          next.delete(key);
          return next;
        });
      }, 600);

      setBoard(newBoard);

      const winner = checkWinner(newBoard);
      if (winner) {
        const winSet = new Set(winner.cells.map(([r, c]) => cellKey(r, c)));
        setWinningCells(winSet);
        setGameState("won");
        setScores((prev) => ({ ...prev, [currentPlayer]: prev[currentPlayer] + 1 }));
      } else if (isBoardFull(newBoard)) {
        setGameState("draw");
      } else {
        setCurrentPlayer(currentPlayer === 1 ? 2 : 1);
      }
    },
    [board, currentPlayer, gameState]
  );

  const newGame = useCallback(() => {
    setBoard(createEmptyBoard());
    setCurrentPlayer(1);
    setGameState("playing");
    setWinningCells(new Set());
    setAnimatingCells(new Set());
  }, []);

  const resetAll = useCallback(() => {
    newGame();
    setScores({ 1: 0, 2: 0 });
  }, [newGame]);

  const getCellClass = (cell: Player | null, row: number, col: number): string => {
    const key = cellKey(row, col);
    const isWin = winningCells.has(key);
    const isAnimating = animatingCells.has(key);

    if (cell === null) {
      const isHovered =
        hoveredCol === col && gameState === "playing" && !isColumnFull(board, col);
      return `w-full aspect-square rounded-full transition-all duration-200 ${
        isHovered ? "bg-blue-800/80" : "bg-blue-900/60"
      }`;
    }

    return [
      "w-full aspect-square rounded-full transition-all duration-300",
      PLAYER_COLORS[cell],
      isWin
        ? `ring-4 ${PLAYER_RING[cell]} shadow-lg ${PLAYER_SHADOW[cell]} animate-win-pulse`
        : `shadow-md ${PLAYER_SHADOW[cell]}`,
      isAnimating ? "animate-drop" : "",
    ]
      .filter(Boolean)
      .join(" ");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 flex flex-col items-center justify-center p-4 gap-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl sm:text-5xl font-black tracking-tight text-white mb-1">
          Connect{" "}
          <span className="bg-gradient-to-r from-red-400 via-yellow-300 to-yellow-400 bg-clip-text text-transparent">
            Four
          </span>
        </h1>
        <p className="text-blue-300/60 text-sm font-medium tracking-widest uppercase">
          Drop it to win it
        </p>
      </div>

      {/* Score Board */}
      <div className="flex gap-4 w-full max-w-sm">
        {([1, 2] as Player[]).map((p) => (
          <div
            key={p}
            className={`flex-1 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 p-4 flex flex-col items-center gap-2 transition-all duration-300 ${
              currentPlayer === p && gameState === "playing"
                ? `ring-2 ${PLAYER_RING[p]} bg-white/10`
                : ""
            }`}
          >
            <div
              className={`w-8 h-8 rounded-full ${PLAYER_COLORS[p]} shadow-lg ${PLAYER_SHADOW[p]}`}
            />
            <span className={`text-xs font-bold uppercase tracking-widest ${PLAYER_TEXT[p]}`}>
              {PLAYER_NAMES[p]}
            </span>
            <span className="text-3xl font-black text-white">{scores[p]}</span>
          </div>
        ))}
      </div>

      {/* Board */}
      <div className="flex flex-col items-center gap-3 w-full max-w-sm sm:max-w-md">
        {/* Column hover indicators */}
        <div className="grid grid-cols-7 gap-1.5 sm:gap-2 w-full px-2">
          {Array.from({ length: COLS }, (_, col) => (
            <div key={col} className="flex items-center justify-center h-8">
              {hoveredCol === col && gameState === "playing" && !isColumnFull(board, col) && (
                <div
                  className={`w-5 h-5 sm:w-6 sm:h-6 rounded-full ${PLAYER_COLORS[currentPlayer]} ${PLAYER_SHADOW[currentPlayer]} shadow-md animate-bounce`}
                />
              )}
            </div>
          ))}
        </div>

        {/* Game board */}
        <div
          className="w-full bg-blue-600 p-2 sm:p-3 rounded-2xl sm:rounded-3xl shadow-2xl shadow-blue-900/50"
          onMouseLeave={() => setHoveredCol(null)}
        >
          <div className="grid grid-cols-7 gap-1.5 sm:gap-2">
            {board.map((row, rowIdx) =>
              row.map((cell, colIdx) => (
                <button
                  key={`${rowIdx}-${colIdx}-${cell ?? "e"}`}
                  onClick={() => handleColumnClick(colIdx)}
                  onMouseEnter={() => setHoveredCol(colIdx)}
                  disabled={gameState !== "playing" || isColumnFull(board, colIdx)}
                  className={`${getCellClass(cell, rowIdx, colIdx)} cursor-pointer disabled:cursor-default focus:outline-none`}
                  aria-label={`Column ${colIdx + 1}`}
                />
              ))
            )}
          </div>
        </div>
      </div>

      {/* Status */}
      <div className="h-14 flex items-center justify-center">
        {gameState === "playing" && (
          <div className="flex items-center gap-3 bg-white/5 border border-white/10 rounded-2xl px-6 py-3 backdrop-blur-sm">
            <div
              className={`w-5 h-5 rounded-full ${PLAYER_COLORS[currentPlayer]} ${PLAYER_SHADOW[currentPlayer]} shadow-md`}
            />
            <span className={`font-semibold ${PLAYER_TEXT[currentPlayer]}`}>
              {PLAYER_NAMES[currentPlayer]}&apos;s turn
            </span>
          </div>
        )}
        {gameState === "won" && (
          <div
            className={`flex items-center gap-3 bg-white/10 border ${PLAYER_BORDER[currentPlayer]} rounded-2xl px-6 py-3 backdrop-blur-sm animate-fade-in`}
          >
            <div
              className={`w-5 h-5 rounded-full ${PLAYER_COLORS[currentPlayer]} ${PLAYER_SHADOW[currentPlayer]} shadow-md`}
            />
            <span className={`font-bold text-lg ${PLAYER_TEXT[currentPlayer]}`}>
              {PLAYER_NAMES[currentPlayer]} wins!
            </span>
            <span className="text-lg">🎉</span>
          </div>
        )}
        {gameState === "draw" && (
          <div className="flex items-center gap-3 bg-white/10 border border-white/20 rounded-2xl px-6 py-3 backdrop-blur-sm animate-fade-in">
            <span className="text-lg">🤝</span>
            <span className="font-bold text-lg text-white">It&apos;s a draw!</span>
          </div>
        )}
      </div>

      {/* Buttons */}
      <div className="flex gap-3">
        <button
          onClick={newGame}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white font-semibold rounded-xl transition-all duration-200 shadow-lg shadow-blue-900/50 hover:shadow-blue-600/40 hover:-translate-y-0.5 active:translate-y-0 text-sm"
        >
          New Game
        </button>
        <button
          onClick={resetAll}
          className="px-6 py-3 bg-white/5 hover:bg-white/10 active:bg-white/5 text-white/70 hover:text-white border border-white/10 hover:border-white/20 font-semibold rounded-xl transition-all duration-200 text-sm"
        >
          Reset Scores
        </button>
      </div>

      {/* Footer */}
      <p className="text-blue-400/30 text-xs mt-2">Click a column to drop your piece</p>
    </div>
  );
}
