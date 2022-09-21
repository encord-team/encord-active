from encord_active.app.common.components import sticky_header
from encord_active.app.common.utils import setup_page
from encord_active.app.data_quality.common import MetricType, load_available_metrics
from encord_active.app.data_quality.sub_pages.explorer import ExplorerPage
from encord_active.app.data_quality.sub_pages.summary import SummaryPage


def summary(metric_type: MetricType):
    def render():
        setup_page()
        page = SummaryPage()

        with sticky_header():
            page.sidebar_options()

        page.build(metric_type.value)

    return render


def explorer(metric_type: MetricType):
    def render():
        setup_page()
        page = ExplorerPage()
        available_metrics = load_available_metrics(metric_type.value)

        with sticky_header():
            selected_df = page.sidebar_options(available_metrics)

        if selected_df is None:
            return

        page.build(selected_df)

    return render
