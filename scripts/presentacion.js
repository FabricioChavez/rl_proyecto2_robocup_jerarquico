const pptxgen = require("pptxgenjs");
const path = require("path");
// Opcional: helper que escribe los colores del tema en el .pptx (ruta en APLICAR_TEMA).
let applyTheme = null;
try { if (process.env.APLICAR_TEMA) ({ applyTheme } = require(process.env.APLICAR_TEMA)); } catch (e) { applyTheme = null; }

const REPO = path.resolve(__dirname, "..");
const FIG = path.join(REPO, "informe", "figuras") + path.sep;
const OUT = path.join(REPO, "informe", "presentacion.pptx");

const THEME = {
  name: "Cancha",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B0B0B", lt1: "FFFFFF", dk2: "0F3D2E", lt2: "EEF3F0",
    accent1: "1E7B4F", accent2: "EB6834", accent3: "2A78D6",
    accent4: "EDA100", accent5: "52514E", accent6: "9CC9B0",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = { verde: "1E7B4F", naranja: "EB6834", azul: "2A78D6", gris: "52514E", rejilla: "E1E0D9", claro: "EEF3F0", oscuro: "0F3D2E" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Control jerárquico en RoboCup 2D";
pres.author = "Fabricio Chávez";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "PORTADA",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.1, w: 11.7, h: 1.6, fontSize: 44, bold: true, color: C.background1, fontFace: "Cambria", valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.4, fontSize: 20, color: C.accent6, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENIDO",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontSize: 32, bold: true, color: C.text2, fontFace: "Cambria", valign: "middle" }, text: "" } },
    { text: { text: "RoboCup 2D · RL jerárquico · 2026-2", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent5, isTextBox: true } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "CIERRE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontSize: 32, bold: true, color: C.background1, fontFace: "Cambria", valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

const ejes = () => ({
  catAxisLabelColor: HEX.gris, valAxisLabelColor: HEX.gris,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 12,
  valGridLine: { color: HEX.rejilla, size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false,
  showLegend: false, showValue: true, dataLabelColor: "0B0B0B",
  valAxisMinVal: 0, valAxisMaxVal: 110, valAxisMajorUnit: 20,
  dataLabelFormatCode: '0.0"%"', valAxisLabelFormatCode: '0"%"',
  titleColor: "0B0B0B", titleFontSize: 14,
});

function tarjeta(slide, x, y, w, h, nombre, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { type: "none" }, objectName: nombre,
  });
}
function stat(slide, x, y, w, numero, etiqueta, detalle, color, nombre) {
  tarjeta(slide, x, y, w, 2.6, nombre);
  slide.addText(numero, { x: x + 0.3, y: y + 0.25, w: w - 0.6, h: 1.2, fontSize: 60, bold: true, fontFace: "Cambria", color, margin: 0, isTextBox: true, objectName: nombre + " número" });
  slide.addText(etiqueta, { x: x + 0.3, y: y + 1.45, w: w - 0.6, h: 0.5, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: nombre + " etiqueta" });
  slide.addText(detalle, { x: x + 0.3, y: y + 1.95, w: w - 0.6, h: 0.5, fontSize: 14, color: C.accent5, margin: 0, isTextBox: true, objectName: nombre + " detalle" });
}

// 1 · Portada
pres.addSection({ title: "Inicio" });
let s = pres.addSlide({ masterName: "PORTADA", sectionTitle: "Inicio" });
s.addText("Control jerárquico en RoboCup 2D", { placeholder: "title" });
s.addText([
  { text: "RL profundo, PBRS e imitación: persecución, conducción y definición", options: { breakLine: true } },
  { text: "Fabricio Chávez · Curso de Aprendizaje por Refuerzo 2026-2", options: { fontSize: 16, color: "FFFFFF" } },
], { placeholder: "body" });
s.addNotes("Ataque 2v1 en el simulador oficial rcssserver. Todo implementado en PyTorch dentro del repositorio.");

// 2 · Resultado principal
pres.addSection({ title: "Resultado" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultado" });
s.addText("47 % de goles: mejora clara, sin llegar al 80 %", { placeholder: "title" });
stat(s, 0.6, 1.7, 3.85, "47,0 %", "Sistema final", "IC 95 %: 40,2 – 53,9 %", C.accent2, "Final");
stat(s, 4.74, 1.7, 3.85, "30,5 %", "Primera jerarquía", "coordinador DAgger humano", C.accent1, "Primera");
stat(s, 8.88, 1.7, 3.85, "0 %", "Política plana PPO", "200k pasos, sin jerarquía", C.accent5, "Plana");
tarjeta(s, 0.6, 4.7, 12.13, 1.9, "Protocolo");
s.addText([
  { text: "Protocolo estándar de prueba", options: { bold: true, breakLine: true, color: "0F3D2E" } },
  { text: "200 episodios independientes por modelo (estados iniciales nuevos, semillas 10000+), política determinista, sin actualizar pesos.", options: { breakLine: true } },
  { text: "Misma condición para todos: 2v1, reglas de recuperación, defensor a potencia 50, arquero activo. Mejora sobre la primera jerarquía: z = 3,4, p < 0,001.", options: {} },
], { x: 0.9, y: 4.85, w: 11.6, h: 1.6, fontSize: 15, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 4, isTextBox: true, objectName: "Texto protocolo" });

// 3 · MDP y jerarquía
pres.addSection({ title: "Método" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Método" });
s.addText("Un ataque 2v1 descompuesto en tres opciones", { placeholder: "title" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.67, y: 1.55, w: 4.0, h: 1.0, rectRadius: 0.12, fill: { color: C.text2 }, line: { type: "none" }, objectName: "Coordinador" });
s.addText([{ text: "Coordinador (Vía A)", options: { bold: true, breakLine: true } }, { text: "3 acciones: elige habilidad", options: { fontSize: 14 } }],
  { x: 4.67, y: 1.55, w: 4.0, h: 1.0, align: "center", valign: "middle", fontSize: 18, color: C.background1, margin: 0, isTextBox: true, objectName: "Texto coordinador" });
const hab = [
  ["Persecución", "12 acciones", "β: balón capturado"],
  ["Conducción", "16 acciones", "β: al pie en x > 36, o perdido"],
  ["Definición", "20 acciones", "β: ya disparó"],
];
hab.forEach((h, i) => {
  const x = 0.6 + i * 4.18;
  const fin = x + 1.9, izq = fin < 6.6;
  s.addShape(pres.shapes.LINE, { x: Math.min(6.67, fin), y: 2.55, w: Math.max(0.01, Math.abs(fin - 6.67)), h: 0.6, line: { color: HEX.gris, width: 1.5, endArrowType: "triangle" }, flipH: izq, objectName: "Flecha " + h[0] });
  tarjeta(s, x, 3.15, 3.8, 1.6, "Habilidad " + h[0], C.background2);
  s.addText([{ text: h[0], options: { bold: true, fontSize: 20, color: "1E7B4F", breakLine: true } }, { text: h[1], options: { breakLine: true } }, { text: h[2], options: { color: "52514E" } }],
    { x: x + 0.25, y: 3.25, w: 3.3, h: 1.4, fontSize: 15, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "Texto " + h[0] });
});
const chips = [
  ["Estado", "34 dimensiones continuas"],
  ["Recompensa", "+10 / −5 / −0,01 por ciclo"],
  ["Horizonte", "200 ciclos, γ = 0,99"],
  ["PBRS", "F = γΦ(s′) − Φ(s)"],
];
chips.forEach((c, i) => {
  const x = 0.6 + i * 3.08;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.2, w: 2.85, h: 1.35, rectRadius: 0.12, fill: { color: "FFFFFF" }, line: { color: HEX.rejilla, width: 1 }, objectName: "Borde " + c[0] });
  s.addText([{ text: c[0], options: { bold: true, color: "0F3D2E", breakLine: true } }, { text: c[1], options: {} }],
    { x: x + 0.2, y: 5.3, w: 2.45, h: 1.15, fontSize: 15, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "Texto chip " + c[0] });
});
s.addNotes("Cada habilidad es una opción con terminación beta(s). El coordinador descuenta con gamma^tau, tau la duración real de la opción, así PBRS sigue telescopando.");

// 4 · Escalera
pres.addSection({ title: "Pilares" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Pilares" });
s.addText("Escalera: el valor converge primero, A2C no llega", { placeholder: "title" });
s.addImage({ path: FIG + "pilar3_curvas.png", x: 0.6, y: 1.55, w: 7.9, h: 3.08, altText: "Curvas de retorno G0 de la escalera de ablación, media ± 1σ en 3 semillas", objectName: "Curvas escalera" });
s.addText("Persecución, 3 semillas, 30k pasos, defensor 50. Retorno G0 sin shaping, media ± 1σ.", { x: 0.6, y: 4.75, w: 7.9, h: 0.4, fontSize: 11, color: C.accent5, margin: 0, isTextBox: true, objectName: "Pie curvas" });
const filas = [
  ["Método", "Pasos al 80 %", "Éxito prueba"],
  ["Q tabular (P1)", "10,0k", "—"],
  ["DQN", "9,0k", "99,8 %"],
  ["Double DQN", "10,3k", "97,0 %"],
  ["A2C", "no llega", "72,0 %"],
  ["PPO", "25,0k", "98,7 %"],
  ["BC humano + PPO", "0", "99,8 %"],
].map((r, i) => r.map((t, j) => ({ text: t, options: { bold: i === 0, color: i === 0 ? "FFFFFF" : "0B0B0B", fill: { color: i === 0 ? "0F3D2E" : (i % 2 ? "FFFFFF" : "EEF3F0") }, align: j ? "right" : "left" } })));
s.addTable(filas, { x: 8.85, y: 1.55, w: 3.9, colW: [1.7, 1.1, 1.1], fontSize: 13, rowH: 0.42, border: { type: "none" }, margin: [0.04, 0.08, 0.04, 0.08], objectName: "Tabla escalera" });
tarjeta(s, 0.6, 5.3, 12.13, 1.4, "Lectura escalera");
s.addText([
  { text: "Fuera de política gana en muestras: ", options: { bold: true } },
  { text: "DQN y DDQN reutilizan el replay y convergen en ~10k pasos; PPO necesita 2,5 veces más. A2C es el más inestable (σ entre semillas 0,80). Arrancar desde BC elimina la exploración y baja σ a 0,01.", options: {} },
], { x: 0.9, y: 5.4, w: 11.6, h: 1.2, fontSize: 15, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "Texto lectura escalera" });

// 5 · PBRS
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Pilares" });
s.addText("PBRS acelera; el shaping ingenuo hace reward hacking", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "Éxito en prueba", labels: ["Sin shaping", "PBRS", "Ingenuo"], values: [64.2, 98.7, 40.5] }], {
  x: 0.6, y: 1.5, w: 6.2, h: 4.9, barDir: "col", barGapWidthPct: 60,
  chartColors: [HEX.gris, HEX.verde, HEX.naranja], ...ejes(), dataLabelPosition: "outEnd",
  showTitle: true, title: "Persecución desde cero: éxito en prueba (600 episodios)", objectName: "Gráfico PBRS",
});
tarjeta(s, 7.2, 1.5, 5.53, 2.3, "Hacking retorno");
s.addText("32 vs 13", { x: 7.5, y: 1.65, w: 5.0, h: 0.9, fontSize: 44, bold: true, fontFace: "Cambria", color: C.accent2, margin: 0, isTextBox: true, objectName: "Número retorno" });
s.addText("Retorno que ve el agente en conducción: el shaping ingenuo lo infla 2,5 veces respecto a PBRS…", { x: 7.5, y: 2.6, w: 5.0, h: 1.0, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "Texto retorno" });
tarjeta(s, 7.2, 4.1, 5.53, 2.3, "Hacking éxito");
s.addText("92,7 % vs 99,2 %", { x: 7.5, y: 4.25, w: 5.0, h: 0.9, fontSize: 40, bold: true, fontFace: "Cambria", color: C.accent1, margin: 0, isTextBox: true, objectName: "Número éxito" });
s.addText("…mientras su éxito real baja (z = 5,7). Cobra por empujar el balón aunque lo pierda. Con PBRS, G con shaping = G − Φ(s₀).", { x: 7.5, y: 5.2, w: 5.0, h: 1.1, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "Texto éxito" });
s.addNotes("Prueba de invariancia: la suma descontada de F telescopa a −Φ(s0) con Φ=0 en terminales, así que Q* con shaping = Q* − Φ(s) y el argmax se conserva.");

// 6 · Imitación
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Pilares" });
s.addText("Imitación: la calidad del experto decide", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "Conducción", labels: ["BC humano", "DAgger humano", "BC experto", "DAgger experto"], values: [13.0, 53.5, 99.5, 100] }], {
  x: 0.6, y: 1.5, w: 5.9, h: 4.3, barDir: "col", barGapWidthPct: 60,
  chartColors: [HEX.azul, HEX.azul, HEX.azul, HEX.azul], ...ejes(), dataLabelPosition: "outEnd",
  showTitle: true, title: "Conducción: éxito en prueba", objectName: "Gráfico conducción",
});
s.addChart(pres.charts.BAR, [{ name: "Definición", labels: ["BC humano", "DAgger humano", "+ Double DQN", "+ PPO zona de tiro"], values: [57.5, 58.0, 83.0, 86.0] }], {
  x: 6.83, y: 1.5, w: 5.9, h: 4.3, barDir: "col", barGapWidthPct: 60,
  chartColors: [HEX.azul, HEX.azul, HEX.naranja, HEX.naranja], ...ejes(), dataLabelPosition: "outEnd",
  showTitle: true, title: "Definición: goles en prueba", objectName: "Gráfico definición",
});
s.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 6.15, w: 0.22, h: 0.22, fill: { color: HEX.azul }, line: { type: "none" }, objectName: "Clave imitación" });
s.addText("Imitación (BC / DAgger)", { x: 1.2, y: 6.07, w: 2.8, h: 0.38, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, objectName: "Texto clave imitación" });
s.addShape(pres.shapes.RECTANGLE, { x: 4.1, y: 6.15, w: 0.22, h: 0.22, fill: { color: HEX.naranja }, line: { type: "none" }, objectName: "Clave RL" });
s.addText("Imitación + ajuste fino RL", { x: 4.4, y: 6.07, w: 3.0, h: 0.38, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, objectName: "Texto clave RL" });
s.addText("Persecución: 100 % en todas las variantes. 200 episodios por barra, defensor 50.", { x: 7.6, y: 6.07, w: 5.1, h: 0.38, fontSize: 12, color: C.accent5, margin: 0, isTextBox: true, objectName: "Nota persecución" });
s.addNotes("El humano reacciona tarde al defensor; el experto programado es consistente y etiqueta todos los estados (DAgger de Ross 2011). En definición, entrenar desde la distribución que la jerarquía entrega (zona de tiro) sube a 86 %.");

// 7 · Vía A
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Pilares" });
s.addText("Vía A: del 30,5 % al 47,0 % de goles", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{
  name: "Goles",
  labels: ["Plana PPO", "Todo + DQN", "Todo + A2C", "Todo + PPO", "Coord. DAgger humano", "Coord. BC experto", "Coord. DAgger experto", "Coord. DAgger + PPO"],
  values: [0.0, 1.0, 16.0, 26.5, 30.5, 41.0, 44.5, 47.0],
}], {
  x: 0.6, y: 1.5, w: 7.6, h: 5.2, barDir: "bar", barGapWidthPct: 45,
  chartColors: [HEX.gris, HEX.naranja, HEX.naranja, HEX.naranja, HEX.azul, HEX.azul, HEX.azul, HEX.naranja],
  ...ejes(), dataLabelPosition: "outEnd", showTitle: true, title: "Goles en 200 episodios (meta: 80 %)", objectName: "Gráfico jerarquía",
});
const puntos = [
  ["Tres cambios suman", "terminación β(s), definición desde la zona de tiro y coordinador BC/DAgger del experto + PPO."],
  ["Cada paso suma poco", "BC 41 → DAgger 44,5 → +PPO 47: monótono, pero no significativo paso a paso con n = 200."],
  ["Ajustar todo con RL empeora", "26,5 / 16 / 1 %: el coordinador deja de ser estacionario."],
];
puntos.forEach((p, i) => {
  tarjeta(s, 8.55, 1.5 + i * 1.78, 4.18, 1.6, "Punto " + (i + 1));
  s.addText([{ text: p[0], options: { bold: true, color: "0F3D2E", breakLine: true } }, { text: p[1], options: {} }],
    { x: 8.8, y: 1.6 + i * 1.78, w: 3.7, h: 1.4, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "Texto punto " + (i + 1) });
});

// 8 · Por qué no 80 %
pres.addSection({ title: "Discusión" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Discusión" });
s.addText("Por qué no llega al 80 %: el traspaso entre habilidades", { placeholder: "title" });
s.addChart(pres.charts.DOUGHNUT, [{ name: "Fin de episodio", labels: ["Gol", "Rival con balón", "Tiempo agotado"], values: [34.0, 44.0, 21.5] }], {
  x: 0.6, y: 1.5, w: 5.4, h: 5.0, holeSize: 55, chartColors: [HEX.verde, HEX.naranja, HEX.gris],
  showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 13, legendColor: HEX.gris,
  showValue: true, showPercent: false, dataLabelFormatCode: '0.0"%"', dataLabelColor: "FFFFFF", dataLabelFontSize: 13, dataLabelFontFace: "+mn-lt",
  showTitle: true, title: "Causa de fin, 200 episodios (diagnóstico)", titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: "0B0B0B",
  objectName: "Gráfico causas",
});
const causas = [
  ["El arquero ataja tras el tiro", "44 %: la última opción activa es persecución, justo después de disparar."],
  ["Conducción se atasca", "21,5 %: el balón queda quieto cerca de x ≈ 38 y nunca se cede a definición."],
  ["Distribution shift entre habilidades", "Definición convierte 86 % aislada, pero en la jerarquía recibe el balón en movimiento, mal orientada y con el defensor encima. El coordinador no es el cuello de botella: su experto de reglas logra lo mismo (42,5 %)."],
];
let y = 1.5;
[1.25, 1.25, 2.3].forEach((h, i) => {
  tarjeta(s, 6.4, y, 6.33, h, "Causa " + (i + 1));
  s.addText([{ text: causas[i][0], options: { bold: true, color: "0F3D2E", breakLine: true } }, { text: causas[i][1], options: {} }],
    { x: 6.65, y: y + 0.1, w: 5.85, h: h - 0.2, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "Texto causa " + (i + 1) });
  y += h + 0.3;
});

// 9 · Cierre
pres.addSection({ title: "Cierre" });
s = pres.addSlide({ masterName: "CIERRE", sectionTitle: "Cierre" });
s.addText("Conclusiones y cómo verlo", { placeholder: "title" });
const conc = [
  "DQN y DDQN son los más eficientes en muestras; PPO es estable pero lento; A2C es el más frágil.",
  "PBRS es seguro y acelera; el shaping ingenuo es peor que no usar shaping.",
  "DAgger con un experto consistente es la palanca más fuerte: conducción pasa de 13 % a 100 %.",
  "La jerarquía llega a 47 %; para el 80 % hay que entrenar definición con los traspasos reales.",
];
s.addText(conc.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < conc.length - 1 } })),
  { x: 0.6, y: 1.5, w: 7.0, h: 4.8, fontSize: 17, color: C.background1, paraSpaceAfter: 12, valign: "top", isTextBox: true, objectName: "Conclusiones" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.95, y: 1.5, w: 4.78, h: 4.8, rectRadius: 0.12, fill: { color: "0B2A20" }, line: { type: "none" }, objectName: "Caja comandos" });
s.addText([
  { text: "Demos en Docker (noVNC)", options: { bold: true, fontSize: 15, color: "9CC9B0", fontFace: "Calibri", breakLine: true } },
  { text: "docker compose build visual", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "docker compose run --rm", options: { breakLine: true } },
  { text: "  --service-ports visual", options: { breakLine: true } },
  { text: "  python proyecto.py demo jerarquico", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "python proyecto.py demo   (lista las 35 variantes)", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Abrir localhost:6080/vnc.html", options: {} },
], { x: 8.2, y: 1.65, w: 4.3, h: 4.5, fontSize: 13, fontFace: "Courier New", color: C.background1, valign: "top", margin: 0, isTextBox: true, objectName: "Texto comandos" });

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (applyTheme) await applyTheme(OUT, THEME);
  console.log("ok", OUT);
})();
