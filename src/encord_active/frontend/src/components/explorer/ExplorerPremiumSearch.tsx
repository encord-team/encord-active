import { useState } from "react";

import { Button, Tooltip, Upload } from "antd";
import Search from "antd/es/input/Search";
import { SearchOutlined, UploadOutlined } from "@ant-design/icons";

export type ExplorerPremiumSearchState = {
  search: string | File | undefined;
  setSearch: (value: string | File | undefined) => void;
  searchLoading: boolean;
};

export function useExplorerPremiumSearch(): {
  premiumSearchState: ExplorerPremiumSearchState;
} {
  const [search, setSearch] = useState<string | File>();

  return {
    premiumSearchState: {
      search,
      setSearch,
      searchLoading: false,
    },
  };
}

export function ExplorerPremiumSearch(props: {
  premiumSearchState: ExplorerPremiumSearchState;
}) {
  const {
    premiumSearchState: { search, setSearch, searchLoading },
  } = props;

  // FIXME: re-add snippet (probably query json - with option to set current filter state to match??)
  return (
    <div className="mr-2 flex h-full items-center gap-2 border-r border-gray-200 px-4">
      <Tooltip overlay="Text Search">
        <Search
          className="explorer-premium-search"
          placeholder="Search Anything"
          onSearch={(value) => setSearch(value || undefined)}
          allowClear
          loading={searchLoading}
          defaultValue={typeof search === "string" ? search : search?.name}
          value={typeof search !== "string" ? search?.name : undefined}
          suffix={
            search === undefined ? null : (
              <Tooltip
                overlay={`Searching: ${
                  typeof search === "string" ? search : search?.name
                }`}
              >
                <SearchOutlined onClick={() => setSearch(undefined)} />
              </Tooltip>
            )
          }
        />
      </Tooltip>
      <Tooltip overlay="Image Search">
        <Upload
          onChange={({ file }) => setSearch(file as unknown as File)}
          beforeUpload={() => false}
          showUploadList={false}
        >
          <Button
            className="border-none shadow-none"
            icon={<UploadOutlined />}
          />
        </Upload>
      </Tooltip>
      {/* <Select */}
      {/*   value={searchMode} */}
      {/*   onChange={(mode) => { */}
      {/*     setSearchMode(mode); */}
      {/*     setSearch(undefined); */}
      {/*   }} */}
      {/*   options={[ */}
      {/*     { */}
      {/*       value: "embedding", */}
      {/*       label: "Embedding Search", */}
      {/*     }, */}
      {/*     { */}
      {/*       value: "query", */}
      {/*       label: "Query Generation", */}
      {/*     }, */}
      {/*   ]} */}
      {/* /> */}
    </div>
  );
}
