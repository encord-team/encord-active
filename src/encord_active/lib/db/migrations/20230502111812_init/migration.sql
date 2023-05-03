-- CreateTable
CREATE TABLE "Tag" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "name" TEXT NOT NULL,
    "scope" TEXT NOT NULL
);

-- CreateTable
CREATE TABLE "ItemTag" (
    "label_hash" TEXT NOT NULL,
    "data_hash" TEXT NOT NULL,
    "frame" INTEGER NOT NULL,
    "object_hash" TEXT NOT NULL,
    "tag_id" INTEGER NOT NULL,

    PRIMARY KEY ("label_hash", "data_hash", "frame", "object_hash", "tag_id"),
    CONSTRAINT "ItemTag_tag_id_fkey" FOREIGN KEY ("tag_id") REFERENCES "Tag" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "DataUnit" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "data_hash" TEXT NOT NULL,
    "data_title" TEXT NOT NULL,
    "frame" INTEGER NOT NULL,
    "location" TEXT NOT NULL,
    "lr_data_hash" TEXT NOT NULL,
    CONSTRAINT "DataUnit_lr_data_hash_fkey" FOREIGN KEY ("lr_data_hash") REFERENCES "LabelRow" ("data_hash") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "LabelRow" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "label_hash" TEXT,
    "data_hash" TEXT NOT NULL,
    "data_title" TEXT NOT NULL,
    "data_type" TEXT NOT NULL,
    "created_at" TEXT NOT NULL,
    "last_edited_at" TEXT NOT NULL,
    "location" TEXT NOT NULL
);

-- CreateIndex
CREATE UNIQUE INDEX "Tag_name_scope_key" ON "Tag"("name", "scope");

-- CreateIndex
CREATE UNIQUE INDEX "DataUnit_data_hash_frame_key" ON "DataUnit"("data_hash", "frame");

-- CreateIndex
CREATE UNIQUE INDEX "LabelRow_label_hash_key" ON "LabelRow"("label_hash");

-- CreateIndex
CREATE UNIQUE INDEX "LabelRow_data_hash_key" ON "LabelRow"("data_hash");
