import { useState } from "react";
import { MdOutlineNavigateBefore, MdOutlineNavigateNext } from "react-icons/md";
import { classy } from "../../helpers/classy";

const pageSizes = [15, 30, 45, 60] as const;
const defaultPageSize = pageSizes[0];
export type PageSize = (typeof pageSizes)[number];

export const usePagination = <T extends any>(items: T[]) => {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState<PageSize>(defaultPageSize);

  const index = page - 1;
  const pageItems = items.slice(index * pageSize, index * pageSize + pageSize);

  return { page, setPage, pageSize, setPageSize, pageItems };
};

export const Pagination = ({
  current,
  pageSize,
  totalItems,
  onChange,
  onChangePageSize,
}: {
  current: number;
  pageSize: number;
  totalItems: number;
  onChange: (to: number) => void;
  onChangePageSize: (size: PageSize) => void;
}) => {
  const prev = current - 1;
  const next = current + 1;

  let totalPages = (totalItems / pageSize) | 0;
  if (totalItems % pageSize !== 0) totalPages++;

  return (
    <div className="inline-flex gap-5">
      <select
        className="select max-w-xs"
        onChange={(event) =>
          onChangePageSize(parseInt(event.target.value) as PageSize)
        }
        defaultValue={pageSize}
      >
        {pageSizes.map((size) => (
          <option key={size}>{size}</option>
        ))}
      </select>
      <div className="btn-group">
        <button
          onClick={() => onChange(prev)}
          className={classy("btn", { "btn-disabled": current === 1 })}
        >
          <MdOutlineNavigateBefore />
        </button>
        {prev > 1 && (
          <>
            <button onClick={() => onChange(1)} className="btn">
              1
            </button>
            <button className="btn btn-disabled">...</button>
          </>
        )}

        {prev > 0 && (
          <button onClick={() => onChange(prev)} className="btn">
            {prev}
          </button>
        )}
        <button className="btn btn-active">{current}</button>
        {next < totalPages && (
          <button onClick={() => onChange(next)} className="btn">
            {next}
          </button>
        )}
        {next <= totalPages && (
          <>
            <button className="btn btn-disabled">...</button>
            <button onClick={() => onChange(totalPages)} className="btn">
              {totalPages}
            </button>
          </>
        )}
        <button
          onClick={() => onChange(next)}
          className={classy("btn", {
            "btn-disabled": !totalPages || current === totalPages,
          })}
        >
          <MdOutlineNavigateNext />
        </button>
      </div>
      <form
        onSubmit={async (event) => {
          event.preventDefault();
          const form = event.target as HTMLFormElement;
          const value = +(form[0] as HTMLInputElement).value;
          onChange(Math.min(1, Math.max(totalPages, value)));
          form.reset();
        }}
      >
        <input type="number" placeholder="Go to page" className="input w-36" />
      </form>
    </div>
  );
};
