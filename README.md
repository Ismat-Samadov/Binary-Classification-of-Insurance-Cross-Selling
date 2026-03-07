# Connect Four

A beautiful, responsive Connect Four game built with Next.js, React, and Tailwind CSS.

## Features

- **2-player local multiplayer** — Red vs Yellow, take turns on the same device
- **Piece drop animation** — smooth gravity-based drop with bounce effect
- **Win detection** — detects horizontal, vertical, and diagonal four-in-a-row
- **Winning highlight** — winning cells pulse and glow when the game ends
- **Score tracking** — persistent score counter across multiple games
- **Column hover preview** — bouncing indicator shows where your piece will land
- **Draw detection** — detects a full board with no winner
- **Fully responsive** — works great on mobile, tablet, and desktop
- **Custom SVG favicon** — Connect Four themed browser icon

## Tech Stack

- [Next.js 16](https://nextjs.org/) — React framework with App Router
- [React 19](https://react.dev/) — UI library
- [Tailwind CSS v4](https://tailwindcss.com/) — utility-first styling
- TypeScript — type safety throughout

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## How to Play

1. **Red** always goes first
2. Click any column to drop your piece into the lowest available row
3. First player to connect **4 in a row** (horizontally, vertically, or diagonally) wins
4. Click **New Game** to start a fresh round (scores are kept)
5. Click **Reset Scores** to clear the score and start completely fresh

## Project Structure

```
src/
  app/
    globals.css       # Global styles & custom keyframe animations
    layout.tsx        # Root layout with metadata & fonts
    page.tsx          # Entry point — renders the game
  components/
    ConnectFour.tsx   # Full game UI & state management
  lib/
    gameLogic.ts      # Pure game logic (board, drop, win detection)
public/
  favicon.svg         # Custom SVG favicon
```

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run start` | Start production server |
| `npm run lint` | Run ESLint |
