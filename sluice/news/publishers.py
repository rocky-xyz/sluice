from urllib.parse import urlsplit

# Article host -> publisher name, spelled as Yahoo Finance spells it, so that the
# same outlet gets one name whichever vendor the article came from.
DOMAIN_PUBLISHERS = {
    "247wallst.com": "24/7 Wall St.",
    "barchart.com": "Barchart",
    "barrons.com": "Barrons.com",
    "fool.com": "Motley Fool",
    "gurufocus.com": "GuruFocus.com",
    "insidermonkey.com": "Insider Monkey",
    "investopedia.com": "Investopedia",
    "investors.com": "Investor's Business Daily",
    "simplywall.st": "Simply Wall St.",
    "stocktwits.com": "Stocktwits",
    "thestreet.com": "TheStreet",
    "tikr.com": "TIKR",
    "trefis.com": "Trefis",
    "zacks.com": "Zacks",
}

# Hosts that carry other outlets' articles: the host says nothing about the publisher.
AGGREGATOR_DOMAINS = ("yahoo.com",)


def publisher_for(url: str | None) -> str | None:
    """Publisher implied by the site an article is hosted on, or None when it implies none.

    Sites missing from the table are named by their host, which still tells one
    outlet from another.
    """
    host = urlsplit(url or "").netloc.lower().removeprefix("www.")
    if not host or any(host == d or host.endswith("." + d) for d in AGGREGATOR_DOMAINS):
        return None
    for domain, name in DOMAIN_PUBLISHERS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return host
