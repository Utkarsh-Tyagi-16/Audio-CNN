export const getColor = (value: number): [number, number, number] => {
  let r, g, b;

  if (value > 0) {
    r = 255 * (1 - value * 0.8);
    g = 255 * (1 - value * 0.5);
    b = 255;
  } else {
    r = 255;
    g = 255 * (1 + value * 0.5);
    b = 255 * (1 + value * 0.8);
  }

  return [Math.round(r), Math.round(g), Math.round(b)];
};
