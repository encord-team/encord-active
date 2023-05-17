import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { FaMagic } from "react-icons/fa";

import { API, Scope, searchTypeOptions, useApi } from "./api";
import { Spin } from "./Spinner";
import { classy } from "../../helpers/classy";
import { useAutoAnimate } from "@formkit/auto-animate/react";

export const Assistant = ({
  scope,
  setResults,
  disabled = false,
}: {
  scope: Scope;
  setResults: (
    args: Awaited<ReturnType<API["searchInProject"]>>["ids"]
  ) => void;
  disabled?: boolean;
}) => {
  const client = useQueryClient();
  const formRef = useRef<HTMLFormElement>(null);
  const [parent, _] = useAutoAnimate();

  const search = useApi().search;

  const { refetch, isFetching, data } = useQuery(
    ["search"],
    ({ signal }) => {
      const { query, type } = formRef.current!;
      if (query.value)
        return search({ scope, query: query.value, type: type.value }, signal);
    },
    { enabled: false }
  );

  useEffect(() => {
    data?.ids && setResults(data.ids);
  }, [data]);

  return (
    <div ref={parent} className="flex flex-col w-full gap-2">
      <form
        ref={formRef}
        className="w-full flex"
        onSubmit={(e) => (
          e.preventDefault(), client.cancelQueries(["search"]), refetch()
        )}
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
              {isFetching ? <Spin /> : <FaMagic className="text-base" />}
            </button>
            <input
              name="query"
              type="text"
              placeholder="Enter a query"
              className="input input-bordered w-full"
              disabled={disabled}
            />
            <select
              name="type"
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
      {!isFetching && data?.snippet && (
        <div className="mockup-code">
          <pre data-prefix="1">
            <code>{data.snippet}</code>
          </pre>
        </div>
      )}
    </div>
  );
};
