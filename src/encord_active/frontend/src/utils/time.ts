export const getReadableDatetime = (datetime: string) => {
  return new Date(datetime).toLocaleString("en-US", {
    hour: "numeric",
    minute: "numeric",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
};
