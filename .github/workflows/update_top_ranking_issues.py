import os
import itertools
from datetime import datetime, timedelta

from github import Github
from github.Issue import Issue
from github.Repository import Repository


def main():
    start_time: datetime = datetime.now()

    # --- Initialization ---
    # GitHub Workflow will pass in the token as an environment variable,
    # but we can place it in our env when running the script locally,
    # for convenience
    local_github_token: str | None = None
    github_token: str | None = (
        local_github_token or os.getenv("GITHUB_ACCESS_TOKEN")
    )
    github = Github(github_token)

    # repository name
    repo_name: str = "pvlib/pvlib-python"
    repository: Repository = github.get_repo(repo_name)

    # Number of top issues to list
    MAX_ISSUES = 19
    TOP_ISSUES_CARD_NUMBER = 2196

    # Rate limiting
    remaining_requests_before: int = github.rate_limiting[0]
    print(f"Remaining requests before: {remaining_requests_before}")

    # --- Actions ---
    # Get sorted issues
    query: str = (
        f'repo:{repository.full_name} is:open is:issue sort:reactions-+1-desc'
    )
    issues = github.search_issues(query)

    # Format
    ranked_issues = []
    # Continuous number generator for the numbered list, starts at 1
    index_generator = itertools.count(1)
    for issue in issues:
        # Don't include the overview card (skip to next iteration)
        if issue.number == TOP_ISSUES_CARD_NUMBER:
            continue

        # Get numbered list item index from generator
        i = next(index_generator)
        if i >= MAX_ISSUES:
            break

        markdown_bullet_point: str = (
            f"{issue.html_url} " +
            f"({issue._rawData['reactions']['+1']} :thumbsup:)"
        )

        markdown_bullet_point = f"{i}. {markdown_bullet_point}"
        ranked_issues.append(markdown_bullet_point)

    # edit top issues
    top_issues_card: Issue = repository.get_issue(
        number=TOP_ISSUES_CARD_NUMBER
    )
    header = "Top Ranking Issues"
    new_body = header + "\n" + "\n".join(ranked_issues)
    top_issues_card.edit(
        body=new_body
    )

    print(top_issues_card.body)

    run_duration: timedelta = datetime.now() - start_time
    print(run_duration)


if __name__ == "__main__":
    main()
