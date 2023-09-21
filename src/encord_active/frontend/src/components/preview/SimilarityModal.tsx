import { List, Modal, Tag } from "antd";
import { useProjectAnalysisSimilaritySearch } from "../../hooks/queries/useProjectAnalysisSimilaritySearch";
import { SimilarityResult } from "../../openapi/api";
import { GalleryCard } from "./GalleryCard";
import { FeatureHashMap } from "../Types";

export function SimilarityModal(props: {
  projectHash: string;
  analysisDomain: "data" | "annotation";
  selectedMetric: { domain: "annotation" | "data"; metric_key: string };
  predictionHash: string | undefined;
  featureHashMap: FeatureHashMap;
  similarityItem: string | undefined;
  onClose: () => void;
  onExpand: (item: string) => void;
  onClick: (item: string) => void;
  onShowSimilar: (item: string) => void;
  selectedItems: ReadonlySet<string>;
}) {
  const {
    onClose,
    similarityItem,
    analysisDomain,
    selectedMetric,
    projectHash,
    predictionHash,
    featureHashMap,
    onExpand,
    onClick,
    onShowSimilar,
    selectedItems,
  } = props;
  const { data: similarItems, isLoading } = useProjectAnalysisSimilaritySearch(
    projectHash,
    analysisDomain,
    similarityItem ?? "",
    { enabled: similarityItem !== undefined }
  );

  return (
    <Modal
      width="95vw"
      title="Similarity Search"
      open={similarityItem !== undefined}
      onCancel={onClose}
      footer={null}
    >
      <List
        dataSource={similarItems ?? []}
        loading={isLoading}
        grid={{}}
        pagination={{ defaultPageSize: 10 }}
        renderItem={(value: SimilarityResult) => (
          <GalleryCard
            projectHash={projectHash}
            predictionHash={predictionHash}
            itemId={value.item}
            key={value.item}
            selected={selectedItems.has(value.item)}
            selectedMetric={selectedMetric}
            onExpand={onExpand}
            onClick={onClick}
            onShowSimilar={onShowSimilar}
            hideExtraAnnotations
            customTags={
              <Tag bordered={false} color="red" className="rounded-xl">
                Distance:{" "}
                <span className="font-bold">{value.similarity.toFixed(5)}</span>
              </Tag>
            }
            featureHashMap={featureHashMap}
            iou={0.5}
          />
        )}
      />
    </Modal>
  );
}
