from queue import Queue
from rich import print

import nodriver as uc

key_root = "root"
key_url = "url"
key_pagination = "pagination"
key_request = "request"
key_selector = "selector"

google = {
    key_root: {
        key_url: "https://www.google.com/search",
    },
    key_pagination: {
        key_request: {
            "params": {
                "q": "python+generators",
            },
            "timeout": 2,
        },
        key_selector: "#pnnext",
    },
    "listing": {},
    "page": {},
}


def build_url(location, element) -> str:
    print(location)

    return f"{location}/{element.href}/"


nodriver = {
    key_root: {
        key_url: "https://ultrafunkamsterdam.github.io/nodriver",
    },
    key_pagination: {
        key_request: {
            "new_tab": True,
        },
        key_selector: "a.next-page",
    },
    "listing": {},
    "page": {},
}


# https://realpython.com/introduction-to-python-generators/
def next_page(property: dict, user_agent):
    assert key_root in property, f"'{key_root}' key is required"
    assert key_url in property[key_root], f"'{key_root}.{key_url}' key is required"
    if key_pagination in property:
        assert (
            key_selector in property[key_pagination]
        ), f"'{key_pagination}.{key_selector}' is required if {key_pagination} is present"

    assert user_agent, "user_agent is required"

    url = property[key_root].get(key_url) or False
    page_request_kwargs = property[key_pagination].get(key_request) or {}
    next_page_selector = property[key_pagination].get(key_selector) or None

    while url:
        resp = user_agent.get(url, **page_request_kwargs)
        print(f"fetching page at {resp.url = }")

        if resp.status_code < 400:
            yield resp

            url = False

            # url = (
            #     False
            #     if next_page_selector is None
            #     else resp.html.find(next_page_selector, first=True)
            # )


async def main():
    print(f"{nodriver = }")
    browser = await uc.start()

    url = nodriver[key_root][key_url]
    next_page_selector = nodriver[key_pagination][key_selector]

    while url:
        tab = await browser.get(url, **nodriver[key_pagination][key_request])
        print(f"{ url = }, { tab = }")

        if next_page_selector:
            next_page_url = await tab.select(next_page_selector)
            if next_page_url:
                url = build_url(url, next_page_url)
            else:
                url = False
        else:
            url = False

    # pages = (page for page in next_page(property, user_agent))

    # for page in next_page(property, user_agent):
    #     print(page.url)


if __name__ == "__main__":
    # since asyncio.run never worked (for me)
    uc.loop().run_until_complete(main())
