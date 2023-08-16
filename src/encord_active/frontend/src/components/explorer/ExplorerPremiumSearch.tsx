import { useState } from "react";
import { FaMagic } from "react-icons/fa";

import { Button, Select, Space } from "antd";
import Search from "antd/es/input/Search";

type SearchMode = "query" | "embedding";

export type ExplorerPremiumSearchState = {
  search: string | undefined;
  setSearch: (value: string | undefined) => void;
  searchLoading: boolean;
  searchMode: SearchMode;
  setSearchMode: (value: SearchMode) => void;
};

export function useExplorerPremiumSearch(): {
  premiumSearchState: ExplorerPremiumSearchState;
} {
  const [search, setSearch] = useState<string>();
  const [searchMode, setSearchMode] = useState<SearchMode>("embedding");

  return {
    premiumSearchState: {
      search,
      setSearch,
      searchLoading: false,
      searchMode,
      setSearchMode,
    },
  };
}

export function ExplorerPremiumSearch(props: {
  premiumSearchState: ExplorerPremiumSearchState;
}) {
  const {
    premiumSearchState: {
      search,
      setSearch,
      searchMode,
      setSearchMode,
      searchLoading,
    },
  } = props;

  // FIXME: re-add snippet (probably query json - with option to set current filter state to match??)
  return (
    <Space.Compact size="large">
      <Search
        onSearch={setSearch}
        onChange={() => setSearch(undefined)}
        loading={searchLoading}
        enterButton={
          <Button className="bg-white px-4">
            <FaMagic
              color={search === undefined ? "red" : "blue"}
              className="bg-white"
            />
          </Button>
        }
      />
      <Select
        value={searchMode}
        onChange={(mode) => {
          setSearchMode(mode);
          setSearch(undefined);
        }}
        options={[
          {
            value: "embedding",
            label: "Embedding Search",
          },
          {
            value: "query",
            label: "Query Generation",
          },
        ]}
      />
    </Space.Compact>
  );
  /*
  return (
    <div ref={parent} className="flex flex-1 flex-col gap-2">
      <form
        className="flex w-full"
        onSubmit={(e) => {
          e.preventDefault();
          const query = e.currentTarget.query satisfies HTMLInputElement;
          const type = e.currentTarget.type satisfies HTMLInputElement;
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
              {isFetching ? (
                <Spin indicator={loadingIndicator} />
              ) : (
                <FaMagic className="text-base" />
              )}
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
  */
}
