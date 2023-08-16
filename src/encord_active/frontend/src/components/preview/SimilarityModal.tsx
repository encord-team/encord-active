import { List, Modal, Tag } from "antd";
import { useProjectAnalysisSimilaritySearch } from "../../hooks/queries/useProjectAnalysisSimilaritySearch";
import { SimilarityResult } from "../../openapi/api";
import { GalleryCard } from "./GalleryCard";

export function SimilarityModal(props: {
  projectHash: string;
  analysisDomain: "data" | "annotation";
  selectedMetric: { domain: "annotation" | "data"; metric_key: string };
  predictionHash: string | undefined;
  similarityItem: string | undefined;
  onClose: () => void;
}) {
  const {
    onClose,
    similarityItem,
    analysisDomain,
    selectedMetric,
    projectHash,
    predictionHash,
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
            selected={false}
            selectedMetric={selectedMetric}
            onExpand={() => undefined}
            onClick={() => undefined}
            onShowSimilar={() => undefined}
            hideExtraAnnotations
            customTags={
              <Tag bordered={false} color="red" className="rounded-xl">
                Similarity -{" "}
                <span className="font-bold">{value.similarity.toFixed(5)}</span>
              </Tag>
            }
          />
        )}
      />
    </Modal>
  );
}
