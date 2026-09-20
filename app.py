"""Streamlit entry point:  streamlit run app.py"""
import streamlit as st

from scouting.dashboard import theme, views

st.set_page_config(page_title="European Football Player Scouting", page_icon=":soccer:", layout="wide")
theme.inject_css()

data = views.load_data_or_stop()

pages = [
    st.Page(lambda: views.scouting_page(data), title="Player scouting", icon=":material/search:",
            url_path="scouting", default=True),
    st.Page(lambda: views.cluster_explorer_page(data), title="Cluster explorer", icon=":material/hub:",
            url_path="clusters"),
    st.Page(lambda: views.player_database_page(data), title="Player database", icon=":material/table_view:",
            url_path="database"),
    st.Page(lambda: views.shortlist_page(data), title="My shortlist", icon=":material/bookmark:",
            url_path="shortlist"),
]
# Lets a button anywhere (a card, the sidebar, the database) switch pages via st.switch_page,
# which needs the exact Page object for a callable-defined page - see views._switch_to.
st.session_state["_nav_pages"] = {"scouting": pages[0], "clusters": pages[1], "database": pages[2], "shortlist": pages[3]}
st.navigation(pages, position="top").run()
