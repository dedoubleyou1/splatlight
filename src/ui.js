// Slider panel built from a declarative list; every control writes into
// `state` and calls onChange so the loop redraws on demand.

export function buildPanel(root, groups, state, onChange) {
  const inputs = {};
  for (const g of groups) {
    const box = document.createElement('fieldset');
    box.innerHTML = `<legend>${g.title}</legend>`;
    for (const c of g.controls) {
      const row = document.createElement('label');
      row.className = 'row';
      const name = document.createElement('span');
      name.textContent = c.label;
      row.append(name);
      let input, out;
      if (c.type === 'select') {
        input = document.createElement('select');
        c.options.forEach((o, i) => input.add(new Option(o, i)));
        input.value = state[c.key];
        input.oninput = () => { state[c.key] = Number(input.value); onChange(c.key); };
      } else if (c.type === 'check') {
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = state[c.key];
        input.oninput = () => { state[c.key] = input.checked; onChange(c.key); };
      } else if (c.type === 'color') {
        input = document.createElement('input');
        input.type = 'color';
        input.value = state[c.key];
        input.oninput = () => { state[c.key] = input.value; onChange(c.key); };
      } else if (c.type === 'button') {
        input = document.createElement('button');
        input.type = 'button';
        input.textContent = c.text;
        input.onclick = () => c.onClick();
      } else {
        input = document.createElement('input');
        input.type = 'range';
        Object.assign(input, { min: c.min, max: c.max, step: c.step });
        input.value = state[c.key];
        out = document.createElement('output');
        const show = () => { out.value = Number(state[c.key]).toFixed(c.digits ?? 2); };
        show();
        input.oninput = () => { state[c.key] = Number(input.value); show(); onChange(c.key); };
      }
      row.append(input);
      if (out) row.append(out);
      if (c.key) inputs[c.key] = input;
      box.append(row);
    }
    root.append(box);
  }
  return inputs;
}
