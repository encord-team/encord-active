export default function getSubsectionNodes(toc, sectionLabel) {
    console.log(sectionLabel);
    let inSubsection = false;
    const subtoc = [];
    for (var i = 0; i < toc.length; i++) {
      const node = toc[i];
      console.log(node);
      if (node.level === 2) {
          if (node.id === sectionLabel) {
              inSubsection = true;
          } else {
              inSubsection = false;
          }
          continue;
      }
      if (inSubsection) {
          subtoc.push(node);
      }
    }
    console.log(subtoc);
    return subtoc;
}

