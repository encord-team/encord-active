export const getReadableDatetime = (datetime: string) =>
  new Date(datetime).toLocaleString("en-US", {
    hour: "numeric",
    minute: "numeric",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
