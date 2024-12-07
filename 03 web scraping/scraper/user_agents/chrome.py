from scraper.user_agents import UserAgent


class Chrome(UserAgent):

    def get(self, url):
        return super().get(url)
