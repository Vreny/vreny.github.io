import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {},
  }),
}

export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(), // В зависимости от версии Quartz, кнопка сворачивания идет в комплекте или управляется через стили
    Component.Explorer({
      title: "Contents",
      folderClickBehavior: "toggle",
      filterFn: (node) => node.name !== "tags",
    }),
  ],
  right: [
    Component.TableOfContents(),
  ],
}

// Копируем правую панель и для списков, чтобы граф и бэклинки не пропадали там
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Search(),
    Component.Darkmode(),
    Component.Explorer({ 
      title: "Contents",
      folderClickBehavior: "toggle",
      filterFn: (node) => node.name !== "tags",
    }),
  ],
  right: [
    Component.Graph(),
    Component.Backlinks(),
  ],
}
