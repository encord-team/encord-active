import { TagList } from "./Tagging";
import { BsCardText } from "react-icons/bs";
import { FaEdit, FaExpand } from "react-icons/fa";
import { MdImageSearch } from "react-icons/md";
import { VscSymbolClass } from "react-icons/vsc";
import { RiUserLine } from "react-icons/ri";
import { QueryAPI } from "../Types";
import { Button, Card, Col, Row, Space, Spin } from "antd";
import { splitId } from "./id";
import { ImageWithPolygons } from "./ImageWithPolygons";

export function ExplorerGalleryItem(props: {
  projectHash: string;
  queryAPI: QueryAPI;
  itemId: string;
  selected: boolean;
  selectedMetric: { domain: "annotation" | "data"; metric_key: string };
  similaritySearchDisabled: boolean;
  onExpand: () => void;
  onShowSimilar: () => void;
  iou?: number;
}) {
  const {
    projectHash,
    queryAPI,
    itemId,
    selected,
    selectedMetric,
    similaritySearchDisabled,
    onExpand,
    onShowSimilar,
    iou,
  } = props;
  const { du_hash, frame, annotation_hash } = splitId(itemId);
  const { data: preview, isLoading } = queryAPI.useProjectItemPreview(
    projectHash,
    du_hash,
    frame,
    annotation_hash
  );
  const { data: info } = queryAPI.useProjectItemDetails(
    projectHash,
    du_hash,
    frame
  );
  const { data: projectSummary } = queryAPI.useProjectSummary(projectHash);
  const projectSummaryForDomain =
    projectSummary == undefined
      ? {}
      : projectSummary[selectedMetric.domain].metrics;
  const { data: projectAnalysisSummary } = queryAPI.useProjectAnalysisSummary(
    projectHash,
    selectedMetric.domain
  );
  /*
  const [metricName, value] = Object.entries(data.metadata.metrics).find(
    ([metric, _]) =>
      metric.toLowerCase() === selectedMetric?.name.toLowerCase(),
  ) || [selectedMetric?.name, ""];
  const [intValue, floatValue] = [parseInt(value), parseFloat(value)];
  const displayValue =
    intValue === floatValue ? intValue : parseFloat(value).toFixed(4);
  const { description } = data.metadata.metrics;
  const { editUrl } = data;
   */
  const metricName = projectSummaryForDomain[selectedMetric.metric_key].title;
  const displayValue =
    info == null ? NaN : info.metrics[selectedMetric.metric_key];
  const editUrl = "FIXME";
  const description = 4;
  const labelClass = "FIXME-LABEL-CLASS";
  const annotator = "FIXME-ANNOTATOR";

  return (
    <Card
      hoverable
      style={{ width: 240 }}
      onClick={() => {}}
      loading={isLoading}
      bodyStyle={{ padding: 0 }}
      cover={
        <label className="group label relative h-full cursor-pointer p-0 not-last:z-10 not-last:opacity-0">
          <input
            name={itemId}
            type="checkbox"
            checked={selected}
            readOnly
            className="peer checkbox absolute left-2 top-2 checked:!opacity-100 group-hover:opacity-100"
          />
          <div className="absolute top-2 flex w-full justify-center gap-1 group-hover:opacity-100">
            <span>{metricName}:</span>
            <span>{displayValue}</span>
          </div>
          <div className="absolute top-7 flex h-5/6 w-full flex-col gap-3 overflow-y-auto p-2 pb-8 group-hover:opacity-100">
            {/*<TagList tags={data.tags} />*/}
            {description && (
              <div className="flex flex-col">
                <div className="inline-flex items-center gap-1">
                  <BsCardText className="text-base" />
                  <span>Description:</span>
                </div>
                <span>{description}</span>
              </div>
            )}
          </div>
          <div className="flex h-full w-full items-center justify-center rounded bg-gray-100 p-1 outline-base-300 checked:transition-none peer-checked:opacity-100 peer-checked:outline  peer-checked:outline-4 peer-checked:outline-offset-[-4px]">
            {preview != null && (
              <ImageWithPolygons
                className="group-hover:opacity-30"
                preview={preview}
              />
            )}
            <div className="absolute top-1 right-1 flex gap-2 opacity-0 group-hover:opacity-100">
              <button
                onClick={(e) => onExpand?.(e)}
                className="btn btn-square z-20"
              >
                <FaExpand />
              </button>
            </div>
          </div>
        </label>
      }
      actions={[
        <Button
          disabled={similaritySearchDisabled}
          onClick={onShowSimilar}
          type="text"
          key="similarity-search"
          icon={<MdImageSearch />}
        />,
        <Button
          disabled={editUrl == null}
          onClick={() =>
            editUrl ? window.open(editUrl.toString(), "_blank") : null
          }
          type="text"
          key="edit"
          icon={<FaEdit />}
        />,
      ]}
    >
      <Row>
        <VscSymbolClass />
        {labelClass}
      </Row>
      <Row>
        <RiUserLine />
        {annotator}
      </Row>
    </Card>
  );
}
