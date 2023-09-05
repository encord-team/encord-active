import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { FaMagic } from "react-icons/fa";

import { API, Filters, Scope, SearchType, searchTypeOptions } from "./api";
import { Spinner } from "./Spinner";
import { classy } from "../../helpers/classy";
import { useAutoAnimate } from "@formkit/auto-animate/react";

type SearchFn = API["searchInProject"];
type ScopelessSearch = Omit<Parameters<SearchFn>[0], "scope" | "filters">;
type Result = Awaited<ReturnType<SearchFn>>;

export const useSearch = (
  scope: Scope,
  filters: Filters,
  searchFn: SearchFn,
) => {
  const client = useQueryClient();

  const [search, setSearch] = useState<ScopelessSearch | undefined>();
  const [result, setResult] = useState<Result | undefined>();

  const { refetch, isFetching, data } = useQuery(
    [scope, "search", search?.type, search?.query],
    ({ signal }) => {
      if (!search?.query) return null;
      client.cancelQueries(["search"]);
      const res = searchFn(
        { scope, filters, query: search.query, type: search.type },
        signal,
      );
      return res;
    },
    { enabled: false },
  );

  useEffect(() => {
    if (search) refetch();
    else setResult(undefined);
  }, [search]);

  useEffect(() => {
    if (data) setResult(data);
  }, [data]);

  return { search, setSearch, result, loading: isFetching };
};

export const Assistant = ({
  defaultSearch,
  isFetching,
  snippet,
  setSearch,
  disabled = false,
}: {
  defaultSearch?: ScopelessSearch;
  isFetching: boolean;
  snippet?: string | null;
  setSearch: (search: ScopelessSearch) => void;
  disabled?: boolean;
}) => {
  const [parent, _] = useAutoAnimate();

  return (
    <div ref={parent} className="flex flex-col gap-2 flex-1">
      <form
        className="w-full flex"
        onSubmit={(e) => {
          e.preventDefault();
          const query = e.currentTarget["query"] satisfies HTMLInputElement;
          const type = e.currentTarget["type"] satisfies HTMLInputElement;
          setSearch({ query: query.value, type: type.value as SearchType });
        }}
      >
        <div
          className={classy("form-control w-full", {
            tooltip: disabled,
          })}
          data-tip="Only available on the hosted version"
        >
          <label className="input-group w-full">
            <button
              className={classy("btn btn-square", {
                "btn-disabled": isFetching || disabled,
              })}
            >
              {isFetching ? <Spinner /> : <FaMagic className="text-base" />}
            </button>
            <input
              name="query"
              type="text"
              key={defaultSearch?.query}
              defaultValue={defaultSearch?.query}
              placeholder="Enter a query"
              className="input input-bordered w-full"
              disabled={disabled}
            />
            <select
              name="type"
              key={defaultSearch?.type}
              defaultValue={defaultSearch?.type}
              className="select select-bordered"
              disabled={disabled}
            >
              {Object.entries(searchTypeOptions).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </form>
      {!isFetching && snippet && (
        <div className="mockup-code">
          <pre data-prefix="1">
            <code>{snippet}</code>
          </pre>
        </div>
      )}
    </div>
  );
};
