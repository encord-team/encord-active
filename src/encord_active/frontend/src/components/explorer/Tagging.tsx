import { Tag } from "antd";
import { TbPolygon } from "react-icons/tb";
import { GrMultiple } from "react-icons/gr";
import { ProjectItemTags, ProjectTag } from "../../openapi/api";
import { Colors } from "../../constants";

export function ItemTags({
  tags: { data, label },
  annotationHash,
  limit,
  className,
}: {
  tags: ProjectItemTags;
  annotationHash?: string;
  limit?: number;
  className?: string;
}) {
  const selectAnnotationTags = annotationHash && label[annotationHash];
  const allAnnotationTags =
    selectAnnotationTags ||
    (Object.values(label).filter(Boolean).flat() as ProjectTag[]);

  const dataTags = data.map((d) => d.name).sort();
  const annotationTags = [...new Set(allAnnotationTags.map((t) => t.name))];

  return (
    <div className={`flex flex-col gap-1 ${className ?? ""}`}>
      {!!data.length && (
        <div className="flex items-center">
          <GrMultiple className="mr-1 inline" />
          <TagList tags={dataTags} limit={limit} />
        </div>
      )}
      {!!annotationTags.length && (
        <div className="flex items-center">
          <TbPolygon className="text-base" />
          <TagList tags={annotationTags} limit={limit} />
        </div>
      )}
    </div>
  );
}

export function TagList(props: { tags: string[]; limit?: number }) {
  const { tags, limit } = props;
  const firstTags = tags.slice(0, limit);
  const remainder = tags.length - firstTags.length;

  return (
    <div className="flex-wrap">
      {firstTags.map((tag) => (
        <Tag key={tag} bordered={false} className="rounded-xl">
          {tag}
        </Tag>
      ))}
      {remainder > 0 && (
        <Tag bordered={false} color={Colors.darkGray} className="rounded-xl">
          + {remainder} more tags
        </Tag>
      )}
    </div>
  );
}
