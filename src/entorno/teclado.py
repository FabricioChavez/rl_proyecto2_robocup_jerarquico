"""Ventana de control del experto humano (tkinter), como un videojuego.

Registra las teclas MANTENIDAS (KeyPress/KeyRelease), así W+D a la vez es una
diagonal. La ventana queda encima de las demás; debe tener el foco para jugar.
Funciona en Windows y dentro de Docker (se ve y se usa desde noVNC).
"""

import time
import tkinter

ALIAS = {
    "up": "w",
    "left": "a",
    "down": "s",
    "right": "d",
    "space": "espacio",
    "escape": "esc",
    "return": "enter",
}


class Control:
    def __init__(self, titulo, ayuda):
        self.tk = tkinter.Tk()
        self.tk.title(titulo)
        self.tk.geometry("620x300+40+40")
        self.tk.attributes("-topmost", True)
        self.modo = tkinter.Label(
            self.tk, font=("DejaVu Sans", 22, "bold"), fg="white", bg="#444444"
        )
        self.modo.pack(fill="x")
        self.texto = tkinter.Label(
            self.tk, font=("DejaVu Sans Mono", 11), justify="left", anchor="w"
        )
        self.texto.pack(fill="both", expand=True, padx=10, pady=6)
        tkinter.Label(
            self.tk, text=ayuda, font=("DejaVu Sans", 9), fg="#333333", justify="left"
        ).pack(side="bottom", padx=8, pady=4)
        self.abajo, self.eventos, self._pendientes = set(), [], {}
        self.cerrado = False
        self.tk.bind_all("<KeyPress>", self._pulsar)
        self.tk.bind_all("<KeyRelease>", self._soltar)
        self.tk.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "cerrado", True))
        self.tk.focus_force()
        self.tk.update()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()

    @staticmethod
    def _tecla(evento):
        key = evento.keysym.lower()
        return ALIAS.get(key, key)

    def _pulsar(self, evento):
        key = self._tecla(evento)
        if key in self._pendientes:
            self.tk.after_cancel(self._pendientes.pop(key))
        if key not in self.abajo:
            self.abajo.add(key)
            self.eventos.append(key)

    def _soltar(self, evento):
        # La autorrepetición de X11 manda soltar/pulsar seguidos: se espera 60 ms.
        key = self._tecla(evento)
        if key in self._pendientes:
            self.tk.after_cancel(self._pendientes[key])
        self._pendientes[key] = self.tk.after(
            60, lambda: (self.abajo.discard(key), self._pendientes.pop(key, None))
        )

    def leer(self):
        """(teclas mantenidas o recién pulsadas, pulsaciones nuevas desde la última lectura)."""
        if self.cerrado:
            return set(), ["q"]
        self.tk.update()
        eventos, self.eventos = self.eventos, []
        return set(self.abajo) | set(eventos), eventos

    def refrescar(self):
        """Procesa la ventana sin consumir las pulsaciones pendientes."""
        if not self.cerrado:
            self.tk.update()

    def estado(self, modo, color, texto):
        if not self.cerrado:
            self.modo.config(text=modo, bg=color)
            self.texto.config(text=texto)
            self.tk.update()

    def esperar(self, texto, validas):
        """Bloquea hasta una tecla de `validas`; devuelve esa tecla, o None para salir."""
        self.estado("PULSA PARA EMPEZAR", "#6250d6", texto)
        self.eventos.clear()
        while True:
            _, eventos = self.leer()
            for key in eventos:
                if key in ("q", "esc"):
                    return None
                if key in validas:
                    return key
            time.sleep(0.03)

    def cerrar(self):
        self.cerrado = True
        if self.tk is not None:
            self.tk.destroy()
            self.tk = None
