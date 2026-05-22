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
  ],
  left: [
    Component.PageTitle(),
    Component.Darkmode(),
    Component.Explorer({
      title: "Contents",
      folderClickBehavior: "toggle",
      filterFn: (node) => node.name !== "tags",
    }),
  ],
  right: [],
}

export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.ArticleTitle()],
  left: [
    Component.PageTitle(),
    Component.Darkmode(),
    Component.Explorer({ title: "Contents" }),
  ],
  right: [],
}
