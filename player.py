import re
from typing import List, Optional
import chess

try:
    from chess_tournament import Player
except ImportError:
    from abc import ABC, abstractmethod
    class Player(ABC):
        def __init__(self, name):
            self.name = name
        def get_move(self, fen):
            pass

FINETUNED_MODEL = 'AnthonyKamau/chess-smollm2-135m'

# ── Evaluation tables 

PIECE_VALUES = {chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900,chess.KING:20000}
PAWN_PST   = [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0]
KNIGHT_PST = [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50]
BISHOP_PST = [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20]
ROOK_PST   = [0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0]
QUEEN_PST  = [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20]
KING_MG    = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20]
KING_EG    = [-50,-40,-30,-20,-20,-30,-40,-50,-30,-20,-10,0,0,-10,-20,-30,-30,-10,20,30,30,20,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,20,30,30,20,-10,-30,-30,-30,0,0,0,0,-30,-30,-50,-30,-30,-30,-30,-30,-30,-50]

PIECE_PST  = {chess.PAWN:PAWN_PST,chess.KNIGHT:KNIGHT_PST,chess.BISHOP:BISHOP_PST,chess.ROOK:ROOK_PST,chess.QUEEN:QUEEN_PST,chess.KING:KING_MG}

def _pst_idx(sq, color):
    f,r = chess.square_file(sq), chess.square_rank(sq)
    return (7-r)*8+f if color==chess.WHITE else r*8+f

def _endgame(board):
    wq = len(board.pieces(chess.QUEEN,chess.WHITE))
    bq = len(board.pieces(chess.QUEEN,chess.BLACK))
    if wq==0 and bq==0: return True
    wm = len(board.pieces(chess.KNIGHT,chess.WHITE))+len(board.pieces(chess.BISHOP,chess.WHITE))
    bm = len(board.pieces(chess.KNIGHT,chess.BLACK))+len(board.pieces(chess.BISHOP,chess.BLACK))
    return wq+bq<=2 and wm+bm<=2

def evaluate(board):
    if board.is_checkmate(): return -200000
    if board.is_stalemate() or board.is_insufficient_material(): return 0
    eg=_endgame(board); score=0
    for color in (chess.WHITE,chess.BLACK):
        sign = 1 if color==chess.WHITE else -1
        for pt in PIECE_VALUES:
            for sq in board.pieces(pt,color):
                score += sign*PIECE_VALUES[pt]
                pst = KING_EG if (pt==chess.KING and eg) else PIECE_PST[pt]
                score += sign*pst[_pst_idx(sq,color)]
        if len(board.pieces(chess.BISHOP,color))>=2: score+=sign*30
    score += 5*sum(1 for _ in board.legal_moves)
    return score if board.turn==chess.WHITE else -score

def _order(board, hints):
    scored = []
    for m in board.legal_moves:
        p = 10000 if m.uci() in hints else 0
        if m.promotion: p+=800+PIECE_VALUES.get(m.promotion,0)
        if board.is_capture(m):
            v=board.piece_at(m.to_square); a=board.piece_at(m.from_square)
            vv=PIECE_VALUES.get(v.piece_type,0) if v else 0
            av=PIECE_VALUES.get(a.piece_type,0) if a else 0
            p+=1000+10*vv-av
        if board.gives_check(m): p+=500
        scored.append((p,m))
    return [m for _,m in sorted(scored,reverse=True)]

def _ab(board, depth, alpha, beta, hints):
    if depth==0 or board.is_game_over(): return evaluate(board)
    for m in _order(board, hints):
        board.push(m)
        s = -_ab(board,depth-1,-beta,-alpha,[])
        board.pop()
        if s>=beta: return beta
        alpha=max(alpha,s)
    return alpha

def _best(board, depth, hints):
    moves = _order(board, hints)
    if not moves: return None
    bm,bs = moves[0],-10000000
    for m in moves:
        board.push(m)
        s = -_ab(board,depth-1,-10000000,10000000,[])
        board.pop()
        if s>bs: bs,bm=s,m
    return bm.uci()

class TransformerPlayer(Player):
    def __init__(self, name='TransformerPlayer'):
        super().__init__(name)
        self._model=self._tokenizer=self._device=None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._device='cuda' if torch.cuda.is_available() else 'cpu'
        dt=torch.float16 if self._device=='cuda' else torch.float32
        self._tokenizer=AutoTokenizer.from_pretrained(FINETUNED_MODEL,trust_remote_code=True)
        self._tokenizer.pad_token=self._tokenizer.eos_token
        self._model=AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL,torch_dtype=dt,device_map='auto',trust_remote_code=True)
        self._model.eval()

    def _hints(self, fen, legal_moves):
        import torch
        if self._model is None: self._load()
        legal_uci = {m.uci() for m in legal_moves}
        board = chess.Board(fen)
        turn  = 'White' if board.turn==chess.WHITE else 'Black'
        shown = [m.uci() for m in legal_moves][:20]
        prompt = (f'You are a chess expert. It is {turn} to move.\n'
                  f'FEN: {fen}\n'
                  f'Legal moves: {", ".join(shown)}\n'
                  f'Best move: ')
        inp = self._tokenizer(prompt,return_tensors='pt').to(self._device)
        hints=[]
        with torch.no_grad():
            out=self._model.generate(**inp,max_new_tokens=6,num_beams=5,
                num_return_sequences=5,early_stopping=True,
                pad_token_id=self._tokenizer.eos_token_id)
        for seq in out:
            dec=self._tokenizer.decode(seq[inp['input_ids'].shape[1]:],skip_special_tokens=True).strip().lower()
            match=re.match(r'([a-h][1-8][a-h][1-8][qrbn]?)',dec)
            if match and match.group(1) in legal_uci and match.group(1) not in hints:
                hints.append(match.group(1))
        return hints

    def get_move(self, fen):
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        if not moves: return None
        p = len(board.piece_map())
        depth = 5 if p<=8 else 4 if p<=16 else 3 if p<=24 else 2
        hints=[]
        try: hints=self._hints(fen,moves)
        except: pass
        try:
            mv=_best(board,depth,hints)
            if mv and chess.Move.from_uci(mv) in board.legal_moves: return mv
        except: pass
        try:
            mv=_best(board,max(2,depth-1),[])
            if mv and chess.Move.from_uci(mv) in board.legal_moves: return mv
        except: pass
        return moves[0].uci()
